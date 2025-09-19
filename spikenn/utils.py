import logging, os
import numpy as np
from numba import int32
from numba.experimental import jitclass



# A class for decision-making 
# In the output layer, neurons are evenly mapped to classes
# A neuron can be explicitly labeled as target or non-target, depending on the involved training mechanisms.
# NOTE: implemented as a jitclass because DecisionMap instance is passed as argument of numba functions in _impl.py
spec = [
    ('n_neurons', int32),
    ('n_nt_neurons', int32),
    ('n_classes', int32),
    ('n_neurons_per_class', int32),
    ('map_class', int32[:]),
    ('map_type', int32[:]),
    ('neuron_mask', int32[:]),
]
@jitclass(spec)
class DecisionMap:
    def __init__(self, n_neurons, n_nt_neurons, n_classes):
        self.n_neurons = n_neurons
        self.n_nt_neurons = n_nt_neurons # Number of non-target neurons
        self.n_classes = n_classes
        # Neurons are evenly mapped to classes
        self.n_neurons_per_class = int(n_neurons / n_classes)
        self.map_class = np.zeros(self.n_neurons, dtype=np.int32) # Class mapping
        self.map_type = np.zeros(self.n_neurons, dtype=np.int32) # Target / non-target mapping
        self.neuron_mask = np.ones(self.n_neurons, dtype=np.int32) # Mask for neurons (1: active, 0: inactive)
        
        for i in range(self.n_neurons):
            self.map_class[i] = int(i/self.n_neurons_per_class) # class indice
            self.map_type[i] = 1 if i%self.n_neurons_per_class >= self.n_nt_neurons else 0 # 1 for target, 0 for non target

    def is_target_neuron(self, n):
        return self.map_type[n] == 1
    
    def is_non_target_neuron(self, n):
        return self.map_type[n] == 0
    
    def get_target_neurons(self, y=None):
        inds = []
        for i in range(self.n_neurons):
            if self.map_type[i] == 1 and (y is None or self.map_class[i] == y): inds.append(i)
        return np.array(inds, dtype=np.int32)

    def get_non_target_neurons(self, y=None, not_y=None):
        inds = []
        for i in range(self.n_neurons):
            if self.map_type[i] == 0 and (y is None or self.map_class[i] == y) and (not_y is None or self.map_class[i] != y): inds.append(i)
        return np.array(inds, dtype=np.int32)
        
    def get_class(self, n):
        return self.map_class[n]



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
        if log_to_file: assert output_path is not None
        self.log_to_file = log_to_file
        self.log_path = None if output_path is None else output_path + ('/' if output_path[-1] != '/' else '')
        if output_path is not None:
            # Compute run version to create a unique file
            version = 0
            while True:
                ok=True
                for file,type in [("log","txt"), ("config","json"), ("weights","npy")]: # TODO: Make it more robust
                    filename = os.path.join(output_path, f"readout_{file}{'' if version == 0 else f'_{version}'}.{type}")
                    if os.path.exists(filename): ok=False
                if ok: break
                else: version += 1
            self.exp_version = '' if version == 0 else f'_{version}'
            if log_to_file:
                self.create_log_dir()
                # Name of the log file
                filename = self.log_path + f'readout_log{self.exp_version}.txt'
                # Init logger
                logging.basicConfig(filename=filename, level=logging.INFO, format='')

    def create_log_dir(self):
        # Create log directory if it does not exist
        if self.log_path is not None and not os.path.exists(self.log_path): os.makedirs(self.log_path)

    def log(self, msg):
        print(msg) # Print to console
        if self.log_to_file: # Print to file
            logging.info(msg)

    def stop(self):
        if self.log_to_file:
            logger = logging.getLogger()
            logger.handlers[0].stream.close()
            logger.removeHandler(logger.handlers[0])