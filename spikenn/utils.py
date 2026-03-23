import logging, os
import numpy as np
from numba import int32, float32
from numba.experimental import jitclass
import wandb

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


# A class for decision-making
# In the output layer, neurons are evenly mapped to classes
# A neuron can be explicitly labeled as target or non-target, depending on the involved training mechanisms.
# NOTE: implemented as a jitclass because DecisionMap instance is passed as argument of numba functions in _impl.py
spec = [
    ("n_neurons", int32),
    ("n_nt_neurons", int32),
    ("n_shared_target_neurons", int32),
    ("n_initial_active_per_class", int32),
    ("n_classes", int32),
    ("n_neurons_per_class", int32),
    ("map_class", int32[:]),
    ("map_type", int32[:]),
    ("neuron_active", int32[:]),
]


@jitclass(spec)
class DecisionMap:
    def __init__(self, n_neurons, n_nt_neurons, n_classes, n_shared_target_neurons=0, n_initial_active_per_class=5):
        self.n_nt_neurons = n_nt_neurons  # Number of non-target neurons
        self.n_shared_target_neurons = n_shared_target_neurons
        self.n_initial_active_per_class = n_initial_active_per_class
        self.n_classes = n_classes
        self.n_neurons = n_neurons

        self.n_neurons_per_class = int(n_neurons / n_classes)
        # Neurons are evenly mapped to classes

        self.map_class = np.empty(self.n_neurons, dtype=np.int32)  # Class mapping
        self.map_type = np.empty(self.n_neurons, dtype=np.int32)  # Target / non-target mapping

        # self.target_times = np.zeros(self.n_neurons, dtype=np.float32)

        # Updated by Wonmo
        # Initialize neuron mask to consider only the first 2 neurons per class at the beginning and increase the number of considered neurons during training
        # first 1 neuron per class is non-target neuron, second neuron
        self.neuron_active = np.zeros(
            self.n_neurons, dtype=np.int32
        )  # Mask for neurons to consider during decision-making (1: consider, 0: ignore)
        initial_active = min(max(self.n_initial_active_per_class, 1), self.n_neurons_per_class)
        for c in range(self.n_classes):
            start_idx = c * self.n_neurons_per_class
            self.neuron_active[start_idx : start_idx + initial_active] = 1
        #######

        for i in range(self.n_neurons):
            self.map_class[i] = int(i / self.n_neurons_per_class)  # class indice
            self.map_type[i] = (
                1 if i % self.n_neurons_per_class >= self.n_nt_neurons else 0
            )  # 1 for target, 0 for non target

        # Mark a subset of target neurons as shared-target pool.
        # map_type: 0=non-target, 1=class-private target, 2=shared target.
        if self.n_shared_target_neurons > 0:
            shared_left = self.n_shared_target_neurons
            # Spread shared targets across classes from each class tail so private
            # low-index target neurons remain available for stable class-specific learning.
            for c in range(self.n_classes):
                if shared_left <= 0:
                    break
                start_idx = c * self.n_neurons_per_class
                end_idx = (c + 1) * self.n_neurons_per_class
                class_target_count = self.n_neurons_per_class - self.n_nt_neurons
                # Reserve at least one private target neuron per class.
                max_shared_in_class = max(class_target_count - 1, 0)
                # Distribute remaining budget approximately uniformly.
                per_class_budget = max(1, int(np.ceil(self.n_shared_target_neurons / self.n_classes)))
                to_mark = min(shared_left, max_shared_in_class, per_class_budget)
                for off in range(to_mark):
                    idx = end_idx - 1 - off
                    if idx < start_idx + self.n_nt_neurons:
                        break
                    if self.map_type[idx] == 1:
                        self.map_type[idx] = 2
                        shared_left -= 1
                        if shared_left <= 0:
                            break

    def is_target_neuron(self, n):
        return self.map_type[n] != 0

    def is_non_target_neuron(self, n):
        return self.map_type[n] == 0

    def get_target_neurons(self, y=None):
        inds = []
        for i in range(self.n_neurons):
            if self.neuron_active[i] != 1:
                continue
            # class-private target
            if self.map_type[i] == 1 and (y is None or self.map_class[i] == y):
                inds.append(i)
            # shared target: available to all classes
            if self.map_type[i] == 2:
                inds.append(i)
        return np.array(inds, dtype=np.int32)

    def get_non_target_neurons(self, y=None, not_y=None):
        inds = []
        for i in range(self.n_neurons):
            if (
                self.map_type[i] == 0
                and (y is None or self.map_class[i] == y)
                and (not_y is None or self.map_class[i] != y)
                and self.neuron_active[i] == 1
            ):
                inds.append(i)
        return np.array(inds, dtype=np.int32)

    # def get_target_mask(self, y=None):
    #     idx = []
    #     for i in range(self.n_neurons):
    #         if self.map_type[i] == 1 and (self.neuron_active[i] == 1): idx.append(i)
    #     return np.array(idx, dtype=np.int32)

    # def set_target_time(self, n_ind, time):
    #     self.target_times[n_ind] = time

    def get_class(self, n):
        return self.map_class[n]

    def get_active(self, c):
        return self.neuron_active[c]


class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.max_acc = 0

    def early_stop(self, acc):
        if acc > self.max_acc:
            self.max_acc = acc
            self.counter = 0
        elif acc <= self.max_acc:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Logger:
    def __init__(self, output_path=None, log_to_file=False):
        if log_to_file:
            assert output_path is not None
        self.log_to_file = log_to_file
        self.log_path = None if output_path is None else output_path + ("/" if output_path[-1] != "/" else "")
        if output_path is not None:
            # Compute run version to create a unique file
            version = 0
            while True:
                ok = True
                for file, type in [("log", "txt"), ("config", "json"), ("weights", "npy")]:  # TODO: Make it more robust
                    filename = os.path.join(
                        output_path, f"readout_{file}{'' if version == 0 else f'_{version}'}.{type}"
                    )
                    if os.path.exists(filename):
                        ok = False
                if ok:
                    break
                else:
                    version += 1
            self.exp_version = "" if version == 0 else f"_{version}"
            if log_to_file:
                self.create_log_dir()
                # Name of the log file
                filename = self.log_path + f"readout_log{self.exp_version}.txt"
                # Init logger
                logging.basicConfig(filename=filename, level=logging.INFO, format="")

    def create_log_dir(self):
        # Create log directory if it does not exist
        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def log(self, msg):
        print(msg)  # Print to console
        if self.log_to_file:  # Print to file
            logging.info(msg)

    def stop(self):
        if self.log_to_file:
            logger = logging.getLogger()
            logger.handlers[0].stream.close()
            logger.removeHandler(logger.handlers[0])


# class wandblogger:
#     def __init__(self, project_name, run_name = None, config = None, output_path = None, log_to_file = False):
#         self.log_to_file = log_to_file
#         self.log_path = output_path + ('/' if output_path and output_path[-1] != '/' else '') if output_path else None

#         # init wandb

#         wandb.init(project=project_name, name= run_name, config= config)

#         if log_to_file and self.log_path:
#             if not os.path.exists(self.log_path):
#                 os.makedirs(self.log_path)
#             filename = os.path.join(self.logt_path, 'readout_log.txt')
#             logging.basicConfig(filename= filename, level=logging.INFO, format = '')

#     def log(self,msg, step = None):
#         print(msg)

#         if isinstance(msg, dict):
#             wandb.log(msg, step = step)
#         else:
#             if self.log_to_file:
#                 logging.info(msg)

#     def log_metrics(self, metrics_dict, step = None):
#         wandb.log(metrics_dict, step = step)

#     def stop(self):
#         wandb.finish()
#         if self.log_to_file:
#             logger = logging.getLogger()
#             for handler in logger.handlers[:]:
#                 handler.close()
#                 logger.removeHandler(handler)


class WandbLogger:
    def __init__(self, project_name=None, run_name=None, config=None, log_to_file=False, output_path=None):
        wandb.init(project=project_name, name=run_name, config=config)
        if log_to_file:
            assert output_path is not None
        self.log_to_file = log_to_file
        self.log_path = None if output_path is None else output_path + ("/" if output_path[-1] != "/" else "")
        self.tb_writer = None
        if output_path is not None:
            # Compute run version to create a unique file
            version = 0
            while True:
                ok = True
                for file, type in [("log", "txt"), ("config", "json"), ("weights", "npy")]:  # TODO: Make it more robust
                    filename = os.path.join(
                        output_path, f"readout_{file}{'' if version == 0 else f'_{version}'}.{type}"
                    )
                    if os.path.exists(filename):
                        ok = False
                if ok:
                    break
                else:
                    version += 1
            self.exp_version = "" if version == 0 else f"_{version}"

            if SummaryWriter is not None:
                tb_dir = os.path.join(output_path, f"tensorboard{self.exp_version}")
                self.tb_writer = SummaryWriter(logdir=tb_dir)

            if log_to_file:
                self.create_log_dir()
                # Name of the log file
                filename = self.log_path + f"readout_log{self.exp_version}.txt"
                # Init logger
                logging.basicConfig(filename=filename, level=logging.INFO, format="")

    def create_log_dir(self):
        # Create log directory if it does not exist
        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def log(self, msg, step=None):
        if isinstance(msg, dict):
            wandb.log(msg, step=step)
            if self.tb_writer is not None:
                self._log_to_tensorboard(msg, step)
        else:
            print(msg)

    def _log_to_tensorboard(self, msg, step=None):
        if step is None:
            return
        for key, value in msg.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.tb_writer.add_scalar(key, float(value), step)

    def logging(self, msg):
        print(msg)  # Print to console
        if self.log_to_file:  # Print to file
            logging.info(msg)

    def stop(self):
        wandb.finish()
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if self.log_to_file:
            logger = logging.getLogger()
            logger.handlers[0].stream.close()
            logger.removeHandler(logger.handlers[0])
