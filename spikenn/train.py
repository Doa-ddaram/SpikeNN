import numpy as np
from spikenn._impl import stdp_multiplicative, stdp_additive, s2stdp, s4nn, spike_sort


# Base STDP optimizer class
class STDPOptimizer:
    
    __slots__ = ('network', 'stdp', 'ap', 'am', 'anti_ap', 'anti_am', 'annealing', 'max_time')

    def __init__(self, network, stdp, ap, am, anti_ap=None, anti_am=None, annealing=1, max_time=1):
        self.network = network # List of SpikingLayer instances 
        self.stdp = stdp # STDP model instance
        self.max_time = max_time # Maximum firing time
        self.ap = ap # Positive learning rate (> 0)
        self.am = am # Negative learning rate (< 0)
        self.anti_ap = -ap if anti_ap is None else anti_ap # < 0, anti STDP is STDP with negative LTP and positive LTD 
        self.anti_am = -am if anti_am is None else anti_am # > 0, anti STDP is STDP with negative LTP and positive LTD 
        self.annealing = annealing # Learning rate annealing

    def anneal(self):
        self.ap *= self.annealing
        self.am *= self.annealing
        self.anti_ap *= self.annealing
        self.anti_am *= self.annealing



# Stabilized Supervised STDP (S2-STDP) optimizer
# NOTE: Can only train the output layer of the network 
class S2STDPOptimizer(STDPOptimizer):
    
    __slots__ = ('t_gap', 'class_inhib', 'use_time_ranges')

    def __init__(self, network, stdp, t_gap, class_inhib, use_time_ranges,
                 ap, am, anti_ap=None, anti_am=None, annealing=1, max_time=1):
        super().__init__(network, stdp, ap, am, anti_ap, anti_am, annealing, max_time)
        self.t_gap = t_gap # Time gap hyperparameter
        self.class_inhib = class_inhib # True for intra-class WTA
        self.use_time_ranges = use_time_ranges # True to use SSTDP instead of S2-STDP
        # self.mask_neuron_type = mask_neuron_type # True to use adaptive neuron pruning
    # NOTE: 
    # The implementation is in a function optimized with Numba to accelerate CPU computations.
    # Hence, code is not very clean, should be reworked...
    # However, Numba is not easily implementable in class methods.
    def __call__(self, outputs, target_ind, decision_map):
        n_layers = len(self.network)

        # Get the weights of each layer inside of a list
        network_weights = []
        for layer_ind in range(n_layers):
            network_weights.append(self.network[layer_ind].weights)

        # Compute the new weights
        stdp_func, stdp_args = self.stdp()
        network_weights = s2stdp(outputs, network_weights, target_ind, decision_map, self.t_gap, self.class_inhib, self.use_time_ranges, self.max_time, self.ap, self.am, self.anti_ap, self.anti_am, stdp_func, stdp_args)

        # Assign the new weights and apply layer constraints
        for layer_ind in range(n_layers):
            self.network[layer_ind].weights = network_weights[layer_ind]
            # Make sure weights are in the layer range
            self.network[layer_ind].clip_weights()
            # Normalize weights of the layer if desired
            self.network[layer_ind].normalize_weights()



# Reward-modulated STDP (R-STDP) optimizer
# NOTE: Can only train the output layer of the network
class RSTDPOptimizer(STDPOptimizer):
    
    __slots__ = ('ap_init', 'am_init', 'anti_ap_init', 'anti_am_init', 'adaptive_lr', 'accuracy_trace')

    def __init__(self, network, stdp, n_classes, ap, am, anti_ap=None, anti_am=None, adaptive_lr=True, annealing=1, max_time=1):
        super().__init__(network, stdp, ap, am, anti_ap, anti_am, annealing, max_time)
        # Keep track of initial values to compute adaptive learning rates 
        self.ap_init = self.ap 
        self.am_init = self.am
        self.anti_ap_init = self.anti_ap
        self.anti_am_init = self.anti_am
        self.adaptive_lr = adaptive_lr
        # Learning rate adapted with accuracy
        mean_acc = 1 / n_classes
        lr_mod = (1 - mean_acc) if adaptive_lr else 1
        anti_lr_mod = mean_acc if adaptive_lr else 1
        self.ap *= lr_mod
        self.am *= lr_mod
        self.anti_ap *= anti_lr_mod
        self.anti_am *= anti_lr_mod
        self.accuracy_trace = []


    # Custom annealing method that can depend on accuracy 
    def anneal(self):
        anneal_factor = self.annealing
        self.ap_init *= anneal_factor
        self.am_init *= anneal_factor
        self.anti_ap_init *= anneal_factor
        self.anti_am_init *= anneal_factor
         # Learning rate adapted with accuracy
        mean_acc = np.mean(self.accuracy_trace)
        lr_mod = (1 - mean_acc) if self.adaptive_lr else 1
        anti_lr_mod = mean_acc if self.adaptive_lr else 1
        self.ap = self.ap_init * lr_mod
        self.am = self.am_init * lr_mod
        self.anti_ap = self.anti_ap_init * anti_lr_mod
        self.anti_am = self.anti_am_init * anti_lr_mod
        self.accuracy_trace = []
        
    
    def __call__(self, outputs, target_ind, decision_map):
        # Unpack outputs of forward pass 
        in_spks, out_spks, mem_pots = outputs[-1]
        
        # Select winner (the neuron firing first with highest membrane potential)
        winner = spike_sort(mem_pots, out_spks)[0]

        # Select the learning rate to apply STDP or anti-STDP
        if decision_map.get_class(winner) == target_ind and decision_map.is_target_neuron(winner):
            lrp, lrm = self.ap, self.am # positive LTP, negative LTD
        else: 
            lrp, lrm = self.anti_ap, self.anti_am # negative LTP, positive LTD

        # Compute weight update for the winner (intra- and inter-class WTA)
        stdp_func, stdp_args = self.stdp()
        self.network[-1].weights[winner] = stdp_func(self.network[-1].weights[winner], in_spks, out_spks[winner], lrp, lrm, 1, *stdp_args)
        # Make sure weights are in the layer range
        self.network[-1].clip_weights()
        # Normalize weights of the layer (if desired)
        self.network[-1].normalize_weights()
            
        # Keep track of accuracy for adaptive lr
        if decision_map.get_class(winner) == target_ind: self.accuracy_trace.append(1)
        else: self.accuracy_trace.append(0)

        

# S4NN optimizer (BP-based)
# NOTE: Can only train the output layer of the network 
class S4NNOptimizer:
    
    __slots__ = ('network', 'lr', 't_gap', 'class_inhib', 'use_time_ranges', 'annealing', 'max_time')

    def __init__(self, network, t_gap, class_inhib, use_time_ranges, lr, annealing=1, max_time=1):
        self.network = network # List of SpikingLayer instances
        self.lr = lr # Positive learning rate
        self.annealing = annealing # Learning rate annealing
        self.max_time = max_time # Maximum firing time
        self.t_gap = t_gap # Time gap hyperparameter
        self.class_inhib = class_inhib # True for intra-class WTA 
        self.use_time_ranges = use_time_ranges # True to use original S4NN 


    def anneal(self):
        self.lr *= self.annealing
        

    # NOTE: 
    # The implementation is in a function optimized with Numba to accelerate CPU computations.
    # Hence, code is not very clean, should be reworked...
    # However, Numba is not easily implementable in class methods.
    def __call__(self, outputs, target_ind, decision_map):
        n_layers = len(self.network)

        # Get the weights of each layer inside of a list
        network_weights = []
        for layer_ind in range(n_layers):
            network_weights.append(self.network[layer_ind].weights)

        # Compute the new weights
        network_weights = s4nn(outputs, network_weights, target_ind, decision_map, self.t_gap, self.class_inhib, self.use_time_ranges, self.max_time, self.lr)

        # Assign the new weights and apply layer constraints
        for layer_ind in range(n_layers):
            self.network[layer_ind].weights = network_weights[layer_ind]
            # Make sure weights are in the layer range
            self.network[layer_ind].clip_weights()
            # Normalize weights of the layer if desired
            self.network[layer_ind].normalize_weights()
            
            

# Base regularizer class
class BaseRegularizer:
    __slots__ = ('layer')

    def __init__(self, layer):
        self.layer = layer


    def on_epoch_start(self):
        pass
        

    def on_epoch_end(self):
        pass
        
        
    def compute(self, y, decision_map):
        pass
        
        
    def __call__(self, outputs, y, decision_map):
        pass
            
            

# Competition regularizer class with two-compartement thresholds
class CompetitionRegularizerTwo(BaseRegularizer):
    __slots__ = ('thr_lr', 'thr_anneal', 'thresholds', 'thr_min', 'thr_max')

    def __init__(self, layer, thr_lr, thr_anneal=1):
        super().__init__(layer)
        self.thr_lr = thr_lr # Threshold learning rate
        self.thr_anneal = thr_anneal # Threshold annealing
        self.thresholds = self.layer.thresholds.copy() # Adaptive thresholds
        self.thr_min = self.layer.thresholds.min() # Minimum threshold values
        self.thr_max = None # Maximum threshold values


    def on_epoch_start(self):
        # Reset to test thresholds
        self.thresholds[:] = self.layer.thresholds[:]
        

    def on_epoch_end(self):
        # Apply annealing
        self.thr_lr *= self.thr_anneal
        
        
    def compute(self, y, decision_map):
        # Get target neurons for the current sample
        n_inds_t = decision_map.get_target_neurons(y)
        # Reset training thresholds for all neurons and
        # set adaptive thresholds to target neurons
        self.layer.thresholds_train[:] = self.layer.thresholds[:]
        self.layer.thresholds_train[n_inds_t] = self.thresholds[n_inds_t]
        
        
    def __call__(self, outputs, y, decision_map):
        if self.thr_lr <= 0: return
        # Unpack outputs of forward pass 
        _, out_spks, mem_pots = outputs[-1]
        # Get target neurons
        n_inds = decision_map.get_target_neurons(y)
        # No competition regulation if < 2 target neurons
        if len(n_inds) <= 1: return
        # Get first firing neuron (winner) and the others (losers)
        sorted_inds = spike_sort(mem_pots[n_inds], out_spks[n_inds])
        winner = n_inds[sorted_inds[0]]
        losers = n_inds[sorted_inds[1:]]
        # Apply competition regulation only if the winner is a target neuron
        if decision_map.is_target_neuron(winner):
            self.thresholds[winner] += self.thr_lr * (len(n_inds)-1) / len(n_inds) 
            self.thresholds[losers] -= self.thr_lr * 1 / len(n_inds)
            self.thresholds = np.clip(self.thresholds, self.thr_min, self.thr_max)
            
            

# Competition regularizer class with one threshold per neuron
class CompetitionRegularizerOne(BaseRegularizer):
    __slots__ = ('thr_lr', 'thr_anneal')

    def __init__(self, layer, thr_lr, thr_anneal=1):
        super().__init__(layer)
        self.thr_lr = thr_lr # Threshold learning rate
        self.thr_anneal = thr_anneal # Threshold annealing
        

    def on_epoch_end(self):
        # Apply annealing
        self.thr_lr *= self.thr_anneal
        # Copy training thresholds to test thresholds 
        self.layer.thresholds = self.layer.thresholds_train.copy()

        
    def __call__(self, outputs, y, decision_map):
        if self.thr_lr <= 0: return
        # Unpack outputs of forward pass 
        _, out_spks, mem_pots = outputs[-1]
        # Get target neurons
        n_inds = decision_map.get_target_neurons(y)
        # No competition regulation if < 2 target neurons
        if len(n_inds) <= 1: return
        # Get first firing neuron (winner) and the others (losers)
        sorted_inds = spike_sort(mem_pots[n_inds], out_spks[n_inds])
        winner = n_inds[sorted_inds[0]]
        losers = n_inds[sorted_inds[1:]]
        # Apply competition regulation only if the winner is a target neuron
        if decision_map.is_target_neuron(winner):
            self.layer.thresholds_train[winner] += self.thr_lr * (len(n_inds)-1) / len(n_inds) 
            self.layer.thresholds_train[losers] -= self.thr_lr * 1 / len(n_inds)



# STDP inferface to store parameters and callable to core function
class MultiplicativeSTDP:
    __slots__ = ('beta', 'w_min', 'w_max')

    def __init__(self, beta, w_min, w_max):
        self.beta = beta
        self.w_min = w_min
        self.w_max = w_max

    # Return the STDP core function and its hyperparameters
    def __call__(self):
        return stdp_multiplicative, (self.w_min, self.w_max, self.beta)



# STDP inferface to store parameters and callable to core function
class AdditiveSTDP:
    __slots__ = ('max_time')
    
    def __init__(self, max_time=np.inf):
        self.max_time = max_time

    # Return the STDP core function and its hyperparameters
    def __call__(self):
        return stdp_additive, (self.max_time,)