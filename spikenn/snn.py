import numpy as np
from spikenn.dataset import DATA_TYPE
from spikenn._impl import weight_norm, forward_fc


# Base spiking layer with IF / LIF neurons
# TODO: Adapt for convolutional architectures 
class SpikingLayer:
    __slots__ = ('shape', 'weights', 'thresholds', 'thresholds_train', 'mask_in', 'mask_out', 'leak_tau', 'w_norm', 'w_min', 'w_max', 'max_time', 'train_mode')

    def __init__(self, shape, firing_threshold, leak_tau, w_init_normal, w_init_mean, w_init_std, w_norm, w_min, w_max, max_time):
        self.shape = shape # (N,M): N is the number of output neurons and M the number of input neurons
        self.leak_tau = leak_tau # For LIF neurons, set to None for IF
        self.w_min = w_min # For weight clipping
        self.w_max = w_max # For weight clipping
        self.max_time = max_time # Maximum firing time
        self.w_norm = None # For weight normalization
        self.mask_in = None # For dropout on the input neurons
        self.mask_out = None # For dropout on the output neurons
        self.train_mode = False # For selecting the thresholds during forward pass

        # Weights initialization
        if w_init_normal: # Normal distribution
            self.weights = np.random.normal(
                loc=w_init_mean, scale=w_init_std,
                size=self.shape).astype(np.float32)
        else: # Uniform distribution
            self.weights = np.random.uniform(
                w_min, w_max, size=self.shape).astype(np.float32)
        if w_norm: # Weight normalization
            self.w_norm = self.weights.sum(1) # normalization factor
            self.normalize_weights()
        
        # Clip weights
        self.clip_weights()

        # Thresholds initialization
        # NOTE: Training thresholds can be modified with regularizers
        self.thresholds = np.ones(self.shape[0], dtype=np.float32) * firing_threshold # Test thresholds
        self.thresholds_train = np.ones(self.shape[0], dtype=np.float32) * firing_threshold # Training Thresholds


    # Clip weights in the range [w_min ; w_max]
    def clip_weights(self):
        self.weights = np.clip(self.weights, self.w_min, self.w_max)


    # Normalize weights to maintain a similar mean among neurons
    def normalize_weights(self):
        if self.w_norm is None: return
        for n_ind in range(self.shape[0]):
            self.weights[n_ind] = weight_norm(self.weights[n_ind], self.w_norm[n_ind])


    # Randomly shut down input neurons with a probability of dropout_rate
    # Set dropout_rate=0 to reset
    def compute_dropout_in(self, dropout_rate):
        if dropout_rate > 0: self.mask_in = np.random.binomial(1, 1-dropout_rate, size=self.shape[1:]).astype(np.uint8)
        else: self.mask_in = None
        

    # Randomly shut down output neurons with a probability of dropout_rate
    # Set dropout_rate=0 to reset
    def compute_dropout_out(self, dropout_rate):
        if dropout_rate > 0: self.mask_out = np.random.binomial(1, 1-dropout_rate, size=self.shape[0]).astype(np.uint8)
        else: self.mask_out = None


    # Convert a dense sample to the SpikingDataset sparse format (see dataset.py)
    # NOTE: This method is useful for multi-layer forward propagation since the output of a layer has a dense format
    def convert_input(self, sample):
        if type(sample) == np.recarray and sample.dtype == DATA_TYPE: 
            return sample
        elif type(sample) == np.ndarray:
            # Sort by timestamps
            sorted_inds = sample.argsort()
            sorted_timestamps = sample[sorted_inds]
            # Mask the values not considered in the sparse representation
            sparse_mask = sorted_timestamps < self.max_time
            # Create a structured numpy array and apply the mask
            sample = np.recarray(np.count_nonzero(sparse_mask), dtype=DATA_TYPE)
            sample.indices = sorted_inds[sparse_mask]
            sample.timestamps = sorted_timestamps[sparse_mask]
            return sample
        else:
            return NotImplementedError(f"Unable to convert input of type {sample.dtype}")


    # Set layer to train mode    
    def train(self):
        self.train_mode = True


    # Set layer to test mode     
    def test(self):
        self.train_mode = False
        # Reset dropout in & out
        self.compute_dropout_in(0)
        self.compute_dropout_out(0)


    # Forward pass
    def __call__(self, sample):
        pass



# Fully-connected spiking layer
class Fc(SpikingLayer):
    __slots__ = ('input_size', 'n_neurons')

    def __init__(self, input_size, n_neurons, firing_threshold, w_init_normal,
                 w_init_mean=0.5, w_init_std=0.01, leak_tau=None,
                 w_norm=False, w_min=0, w_max=1, max_time=1):
        super().__init__((n_neurons, input_size), firing_threshold, leak_tau, w_init_normal, w_init_mean, w_init_std, w_norm, w_min, w_max, max_time)
        self.input_size = input_size
        self.n_neurons = n_neurons

    # NOTE: The implementation is in a function optimized with Numba to accelerate CPU computations.
    def __call__(self, sample):
        # Convert dense sample to SpikingDataset format (needed for multi-layer networks only)
        sample = self.convert_input(sample)
        # Select the employed thresholds
        thresholds = self.thresholds_train if self.train_mode else self.thresholds
        return forward_fc(
            input_size=self.input_size, 
            indices=sample.indices,
            timestamps=sample.timestamps,
            weights=self.weights,
            thresholds=thresholds,
            leak_tau=self.leak_tau,
            mask_in=self.mask_in,
            mask_out=self.mask_out,
            max_time=self.max_time
        )