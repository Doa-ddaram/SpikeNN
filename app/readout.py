import sys
import numpy as np
from tqdm import tqdm
from spikenn.snn import Fc
from spikenn.train import S2STDPOptimizer, RSTDPOptimizer, S4NNOptimizer, AdditiveSTDP, MultiplicativeSTDP, BaseRegularizer, CompetitionRegularizerTwo, CompetitionRegularizerOne
from spikenn.utils import DecisionMap, Logger, EarlyStopper
# from spikenn._impl import spike_sort, mask_neuron
from spikenn._impl import spike_sort

np.set_printoptions(suppress=True) # Remove scientific notation


# A flexible fully-connected SNN model for classification, featuring first-spike coding, single-spike IF/LIF neurons, STDP-based supervised learning rule. 
# NOTE: It is possible to define a multi-layer architecture but the current code does not support multi-layer training
class Readout:

    def __init__(self, n_classes, network, optimizer, regularizer, config, logger=Logger()):
        self.network = network # List of SpikingLayer instances
        self.optimizer = optimizer # STDP-based optimizer instance
        self.regularizer = regularizer # Regularizer instance
        self.config = config # Dict of training parameters
        self.logger = logger # Logger instance

        # Decision map in the output layer
        # Neurons are evenly mapped to classes
        n_neurons = network[-1].n_neurons
        n_nt_neurons = self.config.get("nt_neurons", 0)
        self.decision_map = DecisionMap(n_neurons, n_nt_neurons, n_classes)
        self.n_classes = self.decision_map.n_classes
        self.n_neurons_per_class = self.decision_map.n_neurons_per_class

        self.full_logs = True
        self.save_stats = True
        

    # Training method
    def fit(self, train_dataset, val_dataset=None, test_dataset=None):
        epochs = self.config["epochs"]
        
        # Early stopper
        early_stop = False # Flag to stop the training
        early_stopper = None
        if self.config["early_stopping"] > 0: early_stopper = EarlyStopper(self.config["early_stopping"])
        
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

        check_list = [0 for i in range(self.network[-1].n_neurons)] 
        
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
            
            # Additional training stats
            # Only on the output layer
            if self.save_stats: 
                cnt = 0
                winning_cnt = np.zeros((self.network[-1].n_neurons), dtype=np.float32)

            # Set layers to train mode
            for layer in self.network: layer.train()
            
            # Prepare regularizer
            self.regularizer.on_epoch_start()
            
            # For each sample
            for x, y in tqdm(train_dataset, total=train_dataset.shape[0], disable=not sys.stdout.isatty()):
                
                # Compute dropout in & out
                for layer in self.network: layer.compute_dropout_in(dropout_in)
                for layer in self.network: layer.compute_dropout_out(dropout_out)

                # Compute regularizer
                self.regularizer.compute(y, self.decision_map)
  
                # Forward pass
                outputs = []
                for layer_ind,layer in enumerate(self.network):
                    # in_spks is a dense array of input spike timestamps (n_in,)
                    # x is a dense array of output spike timestamps (n_out,)
                    # mem_pots is a dense array of membrane potentials at spike time (n_out,)
                    in_spks, x, mem_pots = layer(x)
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

                # Training step
                self.regularizer(outputs, y, self.decision_map)
                self.optimizer(outputs, y, self.decision_map)
                
                # Stats when multiple neurons per class (i.e. with NCGs)
                for c in range(self.n_classes):
                    n_idx = np.arange(c* self.n_neurons_per_class, (c+1)*self.n_neurons_per_class)
                    winner = n_idx[spike_sort(mem_pots[n_idx], x[n_idx])[0]]
                    if c == y: 
                        target_updates[winner] += 1
                        
                        ## Updated by Wonmo
                        # Activate next neuron if the current winner neuron has been updated more than 1000 times
                        if epoch % 5 == 4 and (target_updates[winner] > 1000 and winner % self.n_neurons_per_class != 0):
                            active_list.append((y, winner))
                            
                            
                        if self.save_stats: t_updates_trace[epoch, cnt, winner] = 1
                    else:
                        ntarget_updates[winner] += 1
                        if self.save_stats: nt_updates_trace[epoch, cnt, winner] = 1
                if self.save_stats: 
                    winning_cnt[w_ind] += 1
                    neuron_prec_trace[epoch, w_ind] += predicted == y
                    if self.n_neurons_per_class > 1 and isinstance(self.regularizer, CompetitionRegularizerTwo): 
                        thresholds_trace[epoch, cnt, :] = self.regularizer.thresholds[:]
                    cnt += 1
                    
            ## Updated by Wonmo
            # Activate neurons that are not active yet based on the active_list collected during the epoch
            if epoch % 5 == 4:
                print("Active set", set(active_list))
                active_list = list(set(active_list))
                for class_idx, neuron_idx in active_list:
                    if check_list[neuron_idx] == 0:
                        self.activate_neuron(class_idx, neuron_idx)
                        check_list[neuron_idx] = 1
                        
            # Save some training info
            if self.save_stats:
                neuron_prec_trace[epoch] /= winning_cnt
                np.save(f"{self.logger.log_path}/neuron_prec_trace.npy", neuron_prec_trace)
                np.save(f"{self.logger.log_path}/thresholds_trace.npy", thresholds_trace)
                np.save(f"{self.logger.log_path}/t_updates_trace.npy", t_updates_trace)
                np.save(f"{self.logger.log_path}/nt_updates_trace.npy", nt_updates_trace)
                np.save(f"{self.logger.log_path}/weights.npy", self.network[-1].weights)

            # Training logs
            train_acc = train_acc / train_dataset.shape[0]
            if self.full_logs:
                # Stats when multiple neurons per class (i.e. with NCGs)  
                if self.n_neurons_per_class > 1 and isinstance(self.regularizer, CompetitionRegularizerTwo):
                    for c in range(self.n_classes):
                        n_idx = np.arange(c*self.n_neurons_per_class, (c+1)*self.n_neurons_per_class)
                        self.logger.log(target_updates[n_idx])
                        self.logger.log(ntarget_updates[n_idx])
                        self.logger.log(np.round(self.regularizer.thresholds[n_idx],0))
                        self.logger.log(self.decision_map.neuron_mask[n_idx])
                        self.logger.log("")
            for i, layer in enumerate(self.network):
                self.logger.log(f"=== Layer {i} ===")
                self.logger.log(f"\tMean weights: {round(layer.weights.mean(),4)} +- {round(layer.weights.std(),4)} (min:{round(layer.weights.min(),3)} ; max:{round(layer.weights.max(),3)})")
                self.logger.log(f"\tMin firing time: {np.mean(min_out_spks[i])} +- {np.std(min_out_spks[i])}")
                self.logger.log(f"\tMean firing time: {np.mean(mean_out_spks[i])} +- {np.std(mean_out_spks[i])}")
            self.logger.log(f"Accuracy on training set after epoch {epoch}: {round(train_acc,4)}")
            
            # Annealing on the learning rates
            self.optimizer.anneal()
            
            # Update regularizer
            self.regularizer.on_epoch_end()

            # Gridsearch early stopping
            if epoch > 2 and train_acc < gridsearch_stop_acc: return None


            ####################################
            ############ VALIDATION ############

            if val_dataset is not None: 
                val_acc = self.predict(val_dataset)
                self.logger.log(f"Accuracy on validation set after epoch {epoch}: {round(val_acc,4)}")
                
                # Gridsearch early stopping
                if epoch > 2 and val_acc < gridsearch_stop_acc: return None
                
                # Early stopping based on validation accuracy
                if early_stopper is not None:
                    early_stop = early_stopper.early_stop(val_acc)
                    if early_stop: 
                        self.logger.log(f"[INFO] Early stopping triggered (max val_acc:{early_stopper.max_acc})")
                        if test_dataset is not None: # Compute final test accuracy after early stopping trigger
                            test_acc = self.predict(test_dataset)
                            self.logger.log(f"Accuracy on test set after epoch {epoch}: {round(test_acc,4)}")
                        break
           
 
            ####################################
            ############### TEST ###############

            if test_dataset is not None:
                test_acc = self.predict(test_dataset)
                self.logger.log(f"Accuracy on test set after epoch {epoch}: {round(test_acc,4)}")


        return (train_acc,
                val_acc if val_dataset is not None else None,
                test_acc if test_dataset is not None else None
        )


    # Test method
    def predict(self, dataset):
        # Set layers to test mode
        for layer in self.network: layer.test()
        acc = 0
        for x, y in dataset:
            # Forward pass
            for layer in self.network:
                _, x, mem_pots = layer(x)
            # Prediction 
            w_ind = spike_sort(mem_pots, x)[0]
            predicted = self.decision_map.get_class(w_ind)
            acc += predicted == y
        acc = acc / dataset.shape[0]            
        return acc

    
    ## Updated by Wonmo
    # Activate the next neuron in the class
    def activate_neuron(self, class_idx, neuron_idx):
        # print(f"neuron {neuron_idx} update too much")
        # print(f"Activating neuron {neuron_idx + 1} of class {class_idx}")
        
        # Avoid activating non-target neurons
        if neuron_idx % self.n_neurons_per_class == 0:
            return
        
        # Count the number of active neurons in the class
        num_active_neurons_in_class = sum(self.decision_map.neuron_mask[class_idx * self.n_neurons_per_class:(class_idx+1)*self.n_neurons_per_class])
        
        # Avoid activating more neurons than available
        if num_active_neurons_in_class >= self.n_neurons_per_class:
            # print(f"neuron {neuron_idx} Cann1ot activate next neuron, all neurons of class {class_idx} are already active")
            return
        
        # Activate the next non-active neuron
        self.decision_map.neuron_mask[class_idx * self.n_neurons_per_class + num_active_neurons_in_class] = 1
        
        
        
    # Create a Readout instance from a dict config
    @classmethod
    def init_from_dict(cls, config, input_shape, n_classes, output_dir, max_time):
        
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
                leak_tau=layer.get('leak_tau', None), # For LIF neurons,
                w_norm=layer["w_norm"],
                w_min=layer["w_min"],
                w_max=layer["w_max"],
                max_time=max_time
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
                class_inhib=config_optim.get('class_inhib', False), # intra-class WTA, use when multiple neurons per class (e.g. with NCGs),
                use_time_ranges=config_optim.get('use_time_ranges', True), # True for original S4NN training,
                lr=config_optim["lr"],
                annealing=config_optim["annealing"],
                max_time=max_time
            )
        # STDP-based rule
        else:
            # STDP model
            stdp_name = config_optim.get("stdp", "additive")
            stdp = None
            if stdp_name == "additive":
                max_time_stdp = max_time if config_optim.get('ignore_silent', False) == True else np.inf
                stdp = AdditiveSTDP(max_time=max_time_stdp)
            elif config_optim["stdp"] == "multiplicative":
                stdp = MultiplicativeSTDP(config_optim["beta"], config_optim["w_min"], config_optim["w_max"])
            else: raise NotImplementedError(f'STDP rule {stdp_name} not implemented.')
            # Rule
            if config_optim["method"] == "rstdp":
                # If learning rates not specified for anti STDP, use the same as STDP
                if "anti_ap" not in config_optim: config_optim["anti_ap"] = -config_optim["ap"]
                if "anti_am" not in config_optim: config_optim["anti_am"] = -config_optim["am"]
                optim = RSTDPOptimizer(
                    network=network,
                    stdp=stdp,
                    n_classes=n_classes,
                    ap=config_optim["ap"],
                    am=config_optim["am"], 
                    anti_ap=config_optim["anti_ap"],
                    anti_am=config_optim["anti_am"],
                    adaptive_lr=config_optim.get('adaptive_lr', True), 
                    annealing=config_optim["annealing"],
                    max_time=max_time
                )
            elif config_optim["method"] == "s2stdp": # ALSO USED FOR SSTDP!!
                use_time_ranges = config_optim.get('use_time_ranges', False) # For SSTDP training, do not use with S2-STDP
                class_inhib = config_optim.get('class_inhib', False) # intra-class WTA, use when multiple neurons per class (e.g. with NCGs)
                optim = S2STDPOptimizer(
                    network=network,
                    stdp=stdp,
                    t_gap=config_optim["t_gap"],
                    class_inhib=class_inhib,
                    use_time_ranges=use_time_ranges,
                    ap=config_optim["ap"],
                    am=config_optim["am"],
                    annealing=config_optim["annealing"],
                    max_time=max_time
                )
            else: raise NotImplementedError(f'Method {config_optim["method"]} not recognized.')
        
        # Init regularizer 
        if "regularizer" in config:
            config_regul = config["regularizer"]
            use_two_thr = config["regularizer"].get('use_two_thr', True)
            if use_two_thr: # Two-compartment threshold adapation
                regularizer = CompetitionRegularizerTwo(layer=network[-1], thr_lr=config_regul["thr_lr"], thr_anneal=config_regul["thr_anneal"])
            else: # Regular threshold adapation
                regularizer = CompetitionRegularizerOne(layer=network[-1], thr_lr=config_regul["thr_lr"], thr_anneal=config_regul["thr_anneal"])
            
        else:
            regularizer = BaseRegularizer(layer=network[-1]) # No regularization

        # Init logger
        log_to_file = output_dir is not None
        logger = Logger(output_dir, log_to_file)

        # Training parameters
        config_trainer = config["trainer"]
        
        return cls(n_classes=n_classes, network=network, optimizer=optim, regularizer=regularizer, config=config_trainer, logger=logger)