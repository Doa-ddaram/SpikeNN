import sys
import numpy as np
from tqdm import tqdm
from spikenn.snn import Fc
from spikenn.train import (
    S2STDPOptimizer,
    RSTDPOptimizer,
    S4NNOptimizer,
    AdditiveSTDP,
    MultiplicativeSTDP,
    BaseRegularizer,
    CompetitionRegularizerTwo,
    CompetitionRegularizerOne,
)


# Updated by Wonmo
# logger changed to wandb
# from spikenn.utils import DecisionMap, Logger, EarlyStopper
from spikenn.utils import DecisionMap, WandbLogger, EarlyStopper
import wandb

from spikenn._impl import spike_sort

np.set_printoptions(suppress=True)  # Remove scientific notation


# A flexible fully-connected SNN model for classification, featuring first-spike coding, single-spike IF/LIF neurons, STDP-based supervised learning rule.
# NOTE: It is possible to define a multi-layer architecture but the current code does not support multi-layer training
class Readout:

    def __init__(self, n_classes, network, optimizer, regularizer, config, logger=None):
        self.network = network  # List of SpikingLayer instances
        self.optimizer = optimizer  # STDP-based optimizer instance
        self.regularizer = regularizer  # Regularizer instance
        self.config = config  # Dict of training parameters
        self.logger = logger  # Logger instance

        # Decision map in the output layer
        # Neurons are evenly mapped to classes
        n_neurons = network[-1].n_neurons
        n_nt_neurons = self.config.get("nt_neurons", 0)
        n_shared_target_neurons = self.config.get("shared_target_neurons", 0)
        n_initial_active_per_class = self.config.get("initial_active_per_class", 5)
        self.decision_map = DecisionMap(
            n_neurons,
            n_nt_neurons,
            n_classes,
            n_shared_target_neurons,
            n_initial_active_per_class,
        )
        self.n_classes = self.decision_map.n_classes
        self.n_neurons_per_class = self.decision_map.n_neurons_per_class

        self.full_logs = True
        self.save_stats = True

    # Training method
    def fit(self, train_dataset, val_dataset=None, test_dataset=None):
        epochs = self.config["epochs"]

        # Dynamic activation and pruning options
        dynamic_cfg = self.config.get("dynamic_activation", {})
        growth_update_threshold = dynamic_cfg.get("update_threshold", 500)
        # Added for reproducible tuning: allow a short warmup and then apply a lower/stricter trigger.
        # This makes growth behavior explicit in config instead of hard-coding one policy.
        growth_after_epoch = dynamic_cfg.get("growth_after_epoch", 0)
        uncertainty_margin = dynamic_cfg.get("uncertainty_margin", 0.0)
        max_active_per_class = dynamic_cfg.get("max_active_per_class", 0)
        # Optional cap on per-epoch growth to avoid sudden activation spikes.
        max_growth_per_class_per_epoch = dynamic_cfg.get("max_growth_per_class_per_epoch", 0)
        max_growth_classes_per_epoch = dynamic_cfg.get("max_growth_classes_per_epoch", 0)
        min_class_difficulty_for_growth = dynamic_cfg.get("min_class_difficulty_for_growth", 0.0)
        class_error_weight = dynamic_cfg.get("class_error_weight", 1.0)
        class_uncertainty_weight = dynamic_cfg.get("class_uncertainty_weight", 1.0)

        pruning_cfg = self.config.get("pruning", {})
        pruning_enabled = pruning_cfg.get("enabled", False)
        pruning_every_n_epochs = max(1, pruning_cfg.get("every_n_epochs", 1))
        pruning_min_target_updates = pruning_cfg.get("min_target_updates", 0)
        # Relative threshold: prune neurons winning < fraction * (class_total / active_count).
        # More robust than an absolute count because it automatically scales with dataset size
        # and the current number of active neurons per class.
        # If 0.0, falls back to the absolute min_target_updates only.
        pruning_min_target_updates_fraction = pruning_cfg.get("min_target_updates_fraction", 0.0)
        pruning_keep_min_active_per_class = pruning_cfg.get("keep_min_active_per_class", 1)
        pruning_class_difficulty_bias = pruning_cfg.get("class_difficulty_bias", 0.0)

        # Early stopper
        early_stop = False  # Flag to stop the training
        early_stopper = None
        if self.config["early_stopping"] > 0:
            early_stopper = EarlyStopper(self.config["early_stopping"])

        # For gridsearch runs
        gridsearch_stop_acc = self.config.get("gridsearch_stop_acc", 0)

        # Dropout in & out
        dropout_in = self.config.get("dropout_in", 0)
        dropout_out = self.config.get("dropout_out", 0)

        # Additional training stats to save for analysis
        # Only on the output layer
        if self.save_stats:
            thresholds_trace = np.zeros((epochs, train_dataset.shape[0], self.network[-1].n_neurons), dtype=np.float32)
            t_updates_trace = np.zeros((epochs, train_dataset.shape[0], self.network[-1].n_neurons), dtype=np.uint8)
            nt_updates_trace = np.zeros((epochs, train_dataset.shape[0], self.network[-1].n_neurons), dtype=np.uint8)
            neuron_prec_trace = np.zeros((epochs, self.network[-1].n_neurons), dtype=np.float32)

        # Training loop
        for epoch in range(epochs):
            ####################################
            ############# TRAINING #############
            active_list = []

            # Stats
            train_acc = 0
            min_out_spks = [[] for i in range(len(self.network))]
            mean_out_spks = [[] for i in range(len(self.network))]
            target_updates = np.zeros(self.network[-1].n_neurons, dtype=np.int32)
            ntarget_updates = np.zeros(self.network[-1].n_neurons, dtype=np.int32)
            newly_activated = []
            class_sample_count = np.zeros(self.n_classes, dtype=np.int32)
            class_correct_count = np.zeros(self.n_classes, dtype=np.int32)
            class_uncertain_count = np.zeros(self.n_classes, dtype=np.int32)

            # Additional training stats
            # Only on the output layer
            if self.save_stats:
                cnt = 0
                winning_cnt = np.zeros((self.network[-1].n_neurons), dtype=np.float32)

            # Set layers to train mode
            for layer in self.network:
                layer.train()

            # Prepare regularizer
            self.regularizer.on_epoch_start()

            # For each sample
            for x, y in tqdm(train_dataset, total=train_dataset.shape[0], disable=not sys.stdout.isatty()):

                # Compute dropout in & out
                for layer in self.network:
                    layer.compute_dropout_in(dropout_in)
                for layer in self.network:
                    layer.compute_dropout_out(dropout_out)

                # Compute regularizer
                self.regularizer.compute(y, self.decision_map)
                # Forward pass
                outputs = []
                for layer_ind, layer in enumerate(self.network):
                    # in_spks is a dense array of input spike timestamps (n_in,)
                    # x is a dense array of output spike timestamps (n_out,)
                    # mem_pots is a dense array of membrane potentials at spike time (n_out,)

                    # Updated by Wonmo
                    # Modified to pass decision map to layer to consider neurons assigned per class in the forward pass
                    in_spks, x, mem_pots = layer(x, self.decision_map)
                    # Save outputs for weight update
                    outputs.append((in_spks, x, mem_pots))
                    # Save stats
                    min_out_spks[layer_ind].append(x.min())
                    mean_out_spks[layer_ind].append(x.mean())

                # Prediction
                # Based on first neuron to fire
                # If several neurons fire at the same time, the one with the highest membrane potential is selected
                w_ind = spike_sort(mem_pots, x)[0]
                predicted = self.decision_map.get_class(w_ind)
                train_acc += predicted == y
                class_sample_count[y] += 1
                class_correct_count[y] += predicted == y

                if uncertainty_margin > 0:
                    is_uncertain = self._is_uncertain_sample(mem_pots, x, uncertainty_margin)
                else:
                    is_uncertain = True
                class_uncertain_count[y] += is_uncertain

                # Training step
                self.regularizer(outputs, y, self.decision_map)
                self.optimizer(outputs, y, self.decision_map)

                # Stats when multiple neurons per class (i.e. with NCGs)
                for c in range(self.n_classes):
                    n_idx = np.arange(c * self.n_neurons_per_class, (c + 1) * self.n_neurons_per_class)
                    winner = n_idx[spike_sort(mem_pots[n_idx], x[n_idx])[0]]
                    if c == y:
                        target_updates[winner] += 1

                        # Updated by Wonmo
                        # Activate next neuron if the current winner neuron has enough target updates.
                        if (
                            is_uncertain
                            and epoch >= growth_after_epoch
                            and target_updates[winner] >= growth_update_threshold
                            and winner % self.n_neurons_per_class != 0
                        ):
                            active_list.append((y, winner))

                        if self.save_stats:
                            t_updates_trace[epoch, cnt, winner] = 1
                    else:
                        ntarget_updates[winner] += 1
                        if self.save_stats:
                            nt_updates_trace[epoch, cnt, winner] = 1
                if self.save_stats:
                    winning_cnt[w_ind] += 1
                    neuron_prec_trace[epoch, w_ind] += predicted == y
                    if self.n_neurons_per_class > 1 and isinstance(self.regularizer, CompetitionRegularizerTwo):
                        thresholds_trace[epoch, cnt, :] = self.regularizer.thresholds[:]
                    cnt += 1

            class_acc = np.divide(
                class_correct_count,
                np.maximum(class_sample_count, 1),
                dtype=np.float32,
            )
            class_uncertainty = np.divide(
                class_uncertain_count,
                np.maximum(class_sample_count, 1),
                dtype=np.float32,
            )
            class_error = 1.0 - class_acc
            total_class_weight = max(class_error_weight + class_uncertainty_weight, 1e-8)
            class_difficulty = (
                class_error_weight * class_error + class_uncertainty_weight * class_uncertainty
            ) / total_class_weight
            class_hardness = np.zeros(self.n_classes, dtype=np.float32)
            difficulty_order = np.argsort(-class_difficulty, kind="stable")
            hardness_denom = max(self.n_classes - 1, 1)
            for rank, class_idx in enumerate(difficulty_order):
                class_hardness[class_idx] = 1.0 - (rank / hardness_denom)

            # Activate neurons that are not active yet based on the active_list collected during the epoch.
            # Keep only the strongest candidate per class, then prioritize harder classes.
            unique_candidates = {}
            for class_idx, neuron_idx in set(active_list):
                score = int(target_updates[neuron_idx])
                prev = unique_candidates.get(class_idx)
                if prev is None or score > prev[1]:
                    unique_candidates[class_idx] = (neuron_idx, score)

            active_before = int(self.decision_map.neuron_active.sum())
            class_growth_count = np.zeros(self.n_classes, dtype=np.int32)
            grew_per_class = np.zeros(self.n_classes, dtype=np.int32)
            selected_growth_classes = 0
            growth_candidates = sorted(
                unique_candidates.items(),
                key=lambda item: (class_difficulty[item[0]], item[1][1]),
                reverse=True,
            )
            for class_idx, candidate in growth_candidates:
                neuron_idx, _ = candidate
                if class_difficulty[class_idx] < min_class_difficulty_for_growth:
                    continue
                if max_growth_classes_per_epoch > 0 and selected_growth_classes >= max_growth_classes_per_epoch:
                    continue
                if (
                    max_growth_per_class_per_epoch > 0
                    and class_growth_count[class_idx] >= max_growth_per_class_per_epoch
                ):
                    continue
                active_idx = self.activate_neuron(class_idx, neuron_idx, max_active_per_class=max_active_per_class)
                if active_idx is not None:
                    newly_activated.append(active_idx)
                    class_growth_count[class_idx] += 1
                    grew_per_class[class_idx] += 1
                    selected_growth_classes += 1

            active_after_growth = int(self.decision_map.neuron_active.sum())

            pruned_count = 0
            pruned_per_class = np.zeros(self.n_classes, dtype=np.int32)
            if pruning_enabled and ((epoch + 1) % pruning_every_n_epochs == 0):
                pruned_count, pruned_per_class = self.prune_neurons(
                    target_updates=target_updates,
                    ntarget_updates=ntarget_updates,
                    keep_min_active_per_class=pruning_keep_min_active_per_class,
                    min_target_updates=pruning_min_target_updates,
                    min_target_updates_fraction=pruning_min_target_updates_fraction,
                    min_target_precision=pruning_cfg.get("min_target_precision", 0.0),
                    max_prune_per_class=pruning_cfg.get("max_prune_per_class", 0),
                    class_hardness=class_hardness,
                    class_difficulty_bias=pruning_class_difficulty_bias,
                    protected_indices=newly_activated,
                )

            active_after_prune = int(self.decision_map.neuron_active.sum())

            # Save some training info
            # if self.save_stats:
            #     neuron_prec_trace[epoch] /= winning_cnt
            #     np.save(f"{self.logger.log_path}/neuron_prec_trace.npy", neuron_prec_trace)
            #     np.save(f"{self.logger.log_path}/thresholds_trace.npy", thresholds_trace)
            #     np.save(f"{self.logger.log_path}/t_updates_trace.npy", t_updates_trace)
            #     np.save(f"{self.logger.log_path}/nt_updates_trace.npy", nt_updates_trace)
            #     np.save(f"{self.logger.log_path}/weights.npy", self.network[-1].weights)

            # Training logs
            train_acc = train_acc / train_dataset.shape[0]

            self.logger.log(f"Epoch {epoch}: Train Acc {train_acc:.4f}")
            if self.n_neurons_per_class > 1:
                active_total = active_after_prune
                active_per_class = [
                    int(
                        self.decision_map.neuron_active[
                            c * self.n_neurons_per_class : (c + 1) * self.n_neurons_per_class
                        ].sum()
                    )
                    for c in range(self.n_classes)
                ]
                self.logger.log(
                    f"[INFO] Active neurons total={active_total}, per-class={active_per_class}, "
                    f"pre={active_before}, post_grow={active_after_growth}, post_prune={active_after_prune}, "
                    f"grew={len(newly_activated)}, pruned={pruned_count}"
                )
                self.logger.log(
                    f"[INFO] Class difficulty={np.round(class_difficulty, 3).tolist()}, "
                    f"hardness={np.round(class_hardness, 3).tolist()}, "
                    f"grew_per_class={grew_per_class.tolist()}, pruned_per_class={pruned_per_class.tolist()}"
                )
            if self.full_logs:
                # Stats when multiple neurons per class (i.e. with NCGs)
                if self.n_neurons_per_class > 1 and isinstance(self.regularizer, CompetitionRegularizerTwo):
                    for c in range(self.n_classes):
                        n_idx = np.arange(c * self.n_neurons_per_class, (c + 1) * self.n_neurons_per_class)
                        self.logger.logging(f"target_updates[{n_idx}]: {target_updates[n_idx]}")
                        self.logger.logging(f"ntarget_updates[{n_idx}]: {ntarget_updates[n_idx]}")
                        self.logger.logging(f"thresholds[{n_idx}]: {np.round(self.regularizer.thresholds[n_idx], 0)}")
                        self.logger.logging("")
            for i, layer in enumerate(self.network):
                self.logger.logging(f"=== Layer {i} ===")

                #### Updated by Wonmo
                # Log the weights of the active neurons in the output layer (i.e. neurons that are assigned to a class in the decision map) to better analyze the training dynamics of the output layer
                active_weights = layer.weights[self.decision_map.neuron_active == 1]
                if active_weights.size > 0:
                    mean_w = active_weights.mean()
                    std_w = active_weights.std()
                    min_w = active_weights.min()
                    max_w = active_weights.max()
                    self.logger.logging(
                        f"\tMean activate weights: {round(mean_w, 4)} +- {round(std_w, 4)} (min:{round(min_w, 3)} ; max:{round(max_w, 3)})"
                    )
                else:
                    self.logger.logging("\tNo active neurons yet.")

                self.logger.logging(f"\tMin firing time: {np.mean(min_out_spks[i])} +- {np.std(min_out_spks[i])}")
                self.logger.logging(f"\tMean firing time: {np.mean(mean_out_spks[i])} +- {np.std(mean_out_spks[i])}")
            self.logger.logging(f"Accuracy on training set after epoch {epoch}: {round(train_acc,4)}")

            # Annealing on the learning rates
            self.optimizer.anneal()

            # Update regularizer
            self.regularizer.on_epoch_end()

            # Gridsearch early stopping
            if epoch > 2 and train_acc < gridsearch_stop_acc:
                return None

            ####################################
            ############ VALIDATION ############

            if val_dataset is not None:
                val_acc = self.predict(val_dataset)
                self.logger.log(f"Accuracy on validation set after epoch {epoch}: {round(val_acc,4)}")

                # Gridsearch early stopping
                if epoch > 2 and val_acc < gridsearch_stop_acc:
                    return None

                # Early stopping based on validation accuracy
                if early_stopper is not None:
                    early_stop = early_stopper.early_stop(val_acc)
                    if early_stop:
                        self.logger.log(f"[INFO] Early stopping triggered (max val_acc:{early_stopper.max_acc})")
                        if test_dataset is not None:  # Compute final test accuracy after early stopping trigger
                            test_acc = self.predict(test_dataset)
                            self.logger.log(f"Accuracy on test set after epoch {epoch}: {round(test_acc,4)}")
                        break

            ####################################
            ############### TEST ###############

            if test_dataset is not None:
                test_acc = self.predict(test_dataset)
                self.logger.log(f"Accuracy on test set after epoch {epoch}: {round(test_acc,4)}")

            ###########################################
            ######### Recoding Stats to Wandb #########

            # --- Active neuron counts for TensorBoard ---
            # These scalars let us track whether the dynamic growth/pruning
            # mechanism is actually firing during training.
            # "neurons/active_total"     : how many neurons are live across all classes
            # "neurons/grew_this_epoch"  : neurons newly activated in this epoch
            # "neurons/pruned_this_epoch": neurons pruned in this epoch
            # "neurons/active_class_{c}" : per-class breakdown (one scalar per class)
            active_neuron_logs = {}
            if self.n_neurons_per_class > 1:
                _active_total = int(self.decision_map.neuron_active.sum())
                active_neuron_logs["neurons/active_total"] = _active_total
                active_neuron_logs["neurons/grew_this_epoch"] = len(newly_activated)
                active_neuron_logs["neurons/pruned_this_epoch"] = pruned_count
                for _c in range(self.n_classes):
                    _s = _c * self.n_neurons_per_class
                    _e = (_c + 1) * self.n_neurons_per_class
                    active_neuron_logs[f"neurons/active_class_{_c}"] = int(self.decision_map.neuron_active[_s:_e].sum())

            log_data = {
                "epoch": epoch,
                "train/accuracy": train_acc,
                "layer/mean_weight": self.network[-1].weights.mean(),
                "layer/weight_dist": wandb.Histogram(self.network[-1].weights),
                "val/accuracy": val_acc if val_dataset is not None else None,
                "test/accuracy": test_acc if test_dataset is not None else None,
                **active_neuron_logs,
            }

            # Update log results for the current epoch
            self.logger.log(log_data, step=epoch + 1)

            if self.full_logs:
                # 1. Prepare dictionary for visualization
                wandb_stats = {}

                for c in range(self.n_classes):
                    n_idx = np.arange(c * self.n_neurons_per_class, (c + 1) * self.n_neurons_per_class)

                    # Visualize Target Updates per class (bar chart)
                    # Can check the number of updates for each neuron as bars in wandb dashboard
                    t_data = [[f"neuron_{i}", val] for i, val in enumerate(target_updates[n_idx])]
                    table_t = wandb.Table(data=t_data, columns=["neuron", "updates"])
                    wandb_stats[f"Class_{c}/Target_Updates"] = wandb.plot.bar(
                        table_t, "neuron", "updates", title=f"Class {c} Target Updates"
                    )

                    # Average threshold values per class (for line graph)
                    # If multiple neurons, compute average or record value of specific representative neuron
                    wandb_stats[f"Class_{c}/Avg_Threshold"] = np.mean(self.regularizer.thresholds[n_idx])

                    # (Optional) Non-Target Updates histogram
                    wandb_stats[f"Class_{c}/NonTarget_Updates_Dist"] = wandb.Histogram(ntarget_updates[n_idx])

                # 2. Send integrated logs
                self.logger.log(wandb_stats)

        return (train_acc, val_acc if val_dataset is not None else None, test_acc if test_dataset is not None else None)

    # Test method
    def predict(self, dataset):
        # Set layers to test mode
        for layer in self.network:
            layer.test()
        acc = 0
        for x, y in dataset:
            # Forward pass
            for layer in self.network:

                # Updated by Wonmo
                # Modified to pass decision map to layer to consider neurons assigned per class in the forward pass
                _, x, mem_pots = layer(x, self.decision_map)

            # Prediction
            w_ind = spike_sort(mem_pots, x)[0]
            predicted = self.decision_map.get_class(w_ind)
            acc += predicted == y
        acc = acc / dataset.shape[0]
        return acc

    # Updated by Wonmo
    # Activate the next neuron in the class
    def activate_neuron(self, class_idx, neuron_idx, max_active_per_class=0):
        # Avoid activating non-target neurons
        if neuron_idx % self.n_neurons_per_class == 0:
            return

        class_start = class_idx * self.n_neurons_per_class
        class_end = class_start + self.n_neurons_per_class
        active_mask = self.decision_map.neuron_active[class_start:class_end]
        num_active_neurons_in_class = int(active_mask.sum())

        # Avoid activating more neurons than available
        if num_active_neurons_in_class >= self.n_neurons_per_class:
            print(
                f"neuron {neuron_idx} Cannot activate next neuron, all neurons of class {class_idx} are already active"
            )
            return None

        if max_active_per_class > 0 and num_active_neurons_in_class >= max_active_per_class:
            return None

        # Find the first truly inactive slot instead of using count-based offset.
        # Count-based offset breaks when pruning removes a middle neuron: the count
        # then lands on an already-active slot, making activate a no-op.
        inactive_positions = np.where(active_mask == 0)[0]
        if len(inactive_positions) == 0:
            return None
        active_idx = class_start + int(inactive_positions[0])
        # Activate the first inactive neuron slot
        self.decision_map.neuron_active[active_idx] = 1

        layer = self.network[-1]
        base_w = layer.weights[neuron_idx].copy()

        # Update weights of the new active neuron by adding small noise to the weights of the current neuron
        layer.weights[active_idx] = base_w + np.random.normal(0, 0.001, size=base_w.shape)

        layer.thresholds[active_idx] = layer.thresholds[neuron_idx]
        layer.thresholds_train[active_idx] = layer.thresholds_train[neuron_idx]
        return active_idx

    def _is_uncertain_sample(self, mem_pots, out_spikes, margin_thr):
        target_inds = self.decision_map.get_target_neurons()
        if target_inds.size < 2:
            return True

        ranked_local = spike_sort(mem_pots[target_inds], out_spikes[target_inds])
        winner = target_inds[ranked_local[0]]
        second = target_inds[ranked_local[1]]
        margin = float(mem_pots[winner] - mem_pots[second])
        return margin <= margin_thr

    def prune_neurons(
        self,
        target_updates,
        ntarget_updates,
        keep_min_active_per_class,
        min_target_updates=0,
        min_target_updates_fraction=0.0,
        min_target_precision=0.0,
        max_prune_per_class=0,
        class_hardness=None,
        class_difficulty_bias=0.0,
        protected_indices=None,
    ):
        """
        Prune under-performing neurons.

        Effective update threshold = max(min_target_updates,
                         min_target_updates_fraction * expected_per_neuron)
        where expected_per_neuron = total_class_updates / active_count.

        Using a relative fraction avoids the failure mode where every neuron exceeds a
        fixed absolute threshold (e.g. min_target_updates=20) because the dataset is large.
        A fraction of 0.3 means "prune any neuron that wins fewer than 30 % of the
        per-neuron average for its class."

        Optional precision gate:
        precision = target_updates / (target_updates + ntarget_updates)
        if precision < min_target_precision, prune aggressively even if update count is high.
        """
        if protected_indices is None:
            protected_indices = []
        protected = set(protected_indices)
        pruned = 0
        pruned_per_class = np.zeros(self.n_classes, dtype=np.int32)

        for class_idx in range(self.n_classes):
            class_start = class_idx * self.n_neurons_per_class
            class_end = (class_idx + 1) * self.n_neurons_per_class
            active_total = int(self.decision_map.neuron_active[class_start:class_end].sum())
            if active_total <= keep_min_active_per_class:
                continue
            pruned_in_class = 0

            # Compute the effective threshold for this class.
            # If a relative fraction is given, use it (scaled to the per-neuron expected count)
            # and take the maximum with the absolute floor so early epochs (low counts) are stable.
            if min_target_updates_fraction > 0.0:
                total_class_updates = int(sum(target_updates[class_start:class_end]))
                expected_per_neuron = total_class_updates / max(active_total, 1)
                effective_threshold = max(
                    min_target_updates,
                    min_target_updates_fraction * expected_per_neuron,
                )
            else:
                effective_threshold = min_target_updates

            hardness = 0.5
            if class_hardness is not None:
                hardness = float(class_hardness[class_idx])
            prune_scale = np.clip(
                1.0 + class_difficulty_bias * (1.0 - 2.0 * hardness),
                0.25,
                3.0,
            )
            effective_threshold *= prune_scale
            class_precision_threshold = np.clip(min_target_precision * prune_scale, 0.0, 0.999)

            candidates = []
            for n_idx in range(class_start, class_end):
                if self.decision_map.neuron_active[n_idx] != 1:
                    continue
                if self.decision_map.is_non_target_neuron(n_idx):
                    continue
                if n_idx in protected:
                    continue
                t_updates = int(target_updates[n_idx])
                nt_updates = int(ntarget_updates[n_idx])
                precision = t_updates / max(t_updates + nt_updates, 1)
                # Sort by worst precision first, then by fewer target updates.
                candidates.append((precision, t_updates, n_idx))

            candidates.sort(key=lambda x: (x[0], x[1]))
            for precision, n_updates, n_idx in candidates:
                # Keep neurons that are both sufficiently updated and sufficiently precise.
                if n_updates >= effective_threshold and precision >= class_precision_threshold:
                    continue
                active_total = int(self.decision_map.neuron_active[class_start:class_end].sum())
                if active_total <= keep_min_active_per_class:
                    break
                self.decision_map.neuron_active[n_idx] = 0
                pruned += 1
                pruned_in_class += 1
                if max_prune_per_class > 0 and pruned_in_class >= max_prune_per_class:
                    break
            pruned_per_class[class_idx] = pruned_in_class

        return pruned, pruned_per_class

    # Create a Readout instance from a dict config
    @classmethod
    def init_from_dict(cls, config, input_shape, n_classes, output_dir, run_name, max_time):

        # Init network
        network = []
        previous_shape = input_shape
        for layer in config["network"]:
            fc = Fc(
                input_size=previous_shape,
                n_neurons=layer["n_neurons"],
                firing_threshold=layer["firing_threshold"],
                w_init_normal=layer["w_init_normal"],
                w_init_mean=layer["w_init_mean"],
                w_init_std=layer["w_init_std"],
                leak_tau=layer.get("leak_tau", None),  # For LIF neurons,
                w_norm=layer["w_norm"],
                w_min=layer["w_min"],
                w_max=layer["w_max"],
                max_time=max_time,
            )
            previous_shape = layer["n_neurons"]
            network.append(fc)

        # Init optimizer
        config_optim = config["optimizer"]
        # BP-based rule
        if config_optim["method"] == "s4nn":
            optim = S4NNOptimizer(
                network=network,
                t_gap=config_optim["t_gap"],
                class_inhib=config_optim.get(
                    "class_inhib", False
                ),  # intra-class WTA, use when multiple neurons per class (e.g. with NCGs),
                use_time_ranges=config_optim.get("use_time_ranges", True),  # True for original S4NN training,
                lr=config_optim["lr"],
                annealing=config_optim["annealing"],
                max_time=max_time,
            )
        # STDP-based rule
        else:
            # STDP model
            stdp_name = config_optim.get("stdp", "additive")
            stdp = None
            if stdp_name == "additive":
                max_time_stdp = max_time if config_optim.get("ignore_silent", False) == True else np.inf
                stdp = AdditiveSTDP(max_time=max_time_stdp)
            elif config_optim["stdp"] == "multiplicative":
                stdp = MultiplicativeSTDP(config_optim["beta"], config_optim["w_min"], config_optim["w_max"])
            else:
                raise NotImplementedError(f"STDP rule {stdp_name} not implemented.")
            # Rule
            if config_optim["method"] == "rstdp":
                # If learning rates not specified for anti STDP, use the same as STDP
                if "anti_ap" not in config_optim:
                    config_optim["anti_ap"] = -config_optim["ap"]
                if "anti_am" not in config_optim:
                    config_optim["anti_am"] = -config_optim["am"]
                optim = RSTDPOptimizer(
                    network=network,
                    stdp=stdp,
                    n_classes=n_classes,
                    ap=config_optim["ap"],
                    am=config_optim["am"],
                    anti_ap=config_optim["anti_ap"],
                    anti_am=config_optim["anti_am"],
                    adaptive_lr=config_optim.get("adaptive_lr", True),
                    annealing=config_optim["annealing"],
                    max_time=max_time,
                )
            elif config_optim["method"] == "s2stdp":  # ALSO USED FOR SSTDP!!
                use_time_ranges = config_optim.get(
                    "use_time_ranges", False
                )  # For SSTDP training, do not use with S2-STDP
                class_inhib = config_optim.get(
                    "class_inhib", False
                )  # intra-class WTA, use when multiple neurons per class (e.g. with NCGs)
                optim = S2STDPOptimizer(
                    network=network,
                    stdp=stdp,
                    t_gap=config_optim["t_gap"],
                    class_inhib=class_inhib,
                    use_time_ranges=use_time_ranges,
                    ap=config_optim["ap"],
                    am=config_optim["am"],
                    annealing=config_optim["annealing"],
                    max_time=max_time,
                )
            else:
                raise NotImplementedError(f'Method {config_optim["method"]} not recognized.')

        # Init regularizer
        if "regularizer" in config:
            config_regul = config["regularizer"]
            use_two_thr = config["regularizer"].get("use_two_thr", True)
            if use_two_thr:  # Two-compartment threshold adapation
                regularizer = CompetitionRegularizerTwo(
                    layer=network[-1], thr_lr=config_regul["thr_lr"], thr_anneal=config_regul["thr_anneal"]
                )
            else:  # Regular threshold adapation
                regularizer = CompetitionRegularizerOne(
                    layer=network[-1], thr_lr=config_regul["thr_lr"], thr_anneal=config_regul["thr_anneal"]
                )

        else:
            regularizer = BaseRegularizer(layer=network[-1])  # No regularization

        # Init logger
        log_to_file = output_dir is not None

        # Integrate necessary config information into Config dict to replace logger with wandb
        Config = {
            "epochs": config["trainer"].get("epochs", 0),
            "limit_neurons": config["network"][-1].get("n_neurons", 0),
            "STDP_method": config_optim["method"],
        }

        logger = WandbLogger(
            project_name="SNN_Project",
            run_name=run_name,
            config=Config,
            output_path=output_dir,
            log_to_file=log_to_file,
        )

        # Training parameters
        config_trainer = config["trainer"]

        return cls(
            n_classes=n_classes,
            network=network,
            optimizer=optim,
            regularizer=regularizer,
            config=config_trainer,
            logger=logger,
        )
