import math
import warnings
import numpy as np
from numba import njit
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


# Weight normalization
def weight_norm(weight, target_norm):
    return weight * (target_norm / np.absolute(weight).sum())



# Sort by spike timestamp and then by membrane potential (at spike time)
# NOTE: Numba is not compatible with np.lexsort((-mem_pots, out_spks))
@njit
def spike_sort(pots, spks):
    idx = np.arange(len(spks))
    for i in range(1, len(idx)):
        key_index = idx[i]
        j = i - 1
        while j >= 0 and (spks[idx[j]] > spks[key_index] or (spks[idx[j]] == spks[key_index] and -pots[idx[j]] > -pots[key_index])):
            idx[j + 1] = idx[j]
            j -= 1
        idx[j + 1] = key_index
    return idx

# @njit
# def delay_learning_rule(delays, in_spks, t_post, t_target, lr_delay, d_min, d_max):
#     n_inputs = delays.shape[0]
#     delays_out = delays.copy()
    
#     for i in range(n_inputs):
#         if in_spks[i] >= 1e10: continue # 스파이크가 없는 경우 skip
        
#         # 실제 도착 시간: t_arrival = t_pre + d
#         t_arrival = in_spks[i] + delays[i]
        
#         # 논문 방식의 핵심: 
#         # t_arrival이 t_target보다 빠르면 delay를 늘리고(늦게 도착하게), 
#         # t_arrival이 t_target보다 늦으면 delay를 줄임(빨리 도착하게).
#         gradient = t_arrival - t_target
#         delays_out[i] -= lr_delay * gradient
        
#         # Clipping (물리적 지연 시간 범위 제한)
#         if delays_out[i] < d_min: delays_out[i] = d_min
#         if delays_out[i] > d_max: delays_out[i] = d_max
        
#     return delays_out

# Forward pass for a fully-connected architecture
@njit
def forward_fc(input_size, indices, timestamps, weights, thresholds, leak_tau, mask_in, mask_out, max_time, delays):
    n_neurons = thresholds.shape[0]
    # Init membrane potentials
    mem_pots = np.zeros(n_neurons, dtype=np.float32)
    # Record input and output spike timestamps (max_time = no spike)
    limit_time = max_time + 1.0
    in_spks = np.ones(input_size, dtype=np.float32) * limit_time
    out_spks = np.ones(n_neurons, dtype=np.float32) * limit_time
    # Record neurons that already fired (single-spike constraint)
    active_neurons = np.ones(n_neurons, dtype=np.uint8)
    # Dropout on the output neurons
    if mask_out is not None:
        for n_ind in range(n_neurons):
            if mask_out[n_ind] == 0: active_neurons[n_ind] = 0
    # if len(timestamps) > 0:
    #     # Keep track of the current timestamp of the simulation 
    #     # to process all input spikes with the same timestamp before computing output spikes
    #     curr_time = timestamps[0] # First timestamp
    #     # Iterate over all input spikes
    #     for i, (ind,time) in enumerate(zip(indices,timestamps)):
    #         # When all input spikes with the same timestamp are processed
    #         # we check for output neurons ready to fire
    #         if curr_time != time or i == len(indices)-1:
    #             # Apply leak according to the time difference between the current and last input spikes
    #             if leak_tau is not None: 
    #                 for n_ind in range(n_neurons):
    #                     if active_neurons[n_ind] ==  1:
    #                         mem_pots[n_ind] *= np.exp(-(time-curr_time)/leak_tau[n_ind])
    #             # Generate output spikes
    #             for n_ind in range(n_neurons):
    #                 if active_neurons[n_ind] == 1 and mem_pots[n_ind] > thresholds[n_ind]:
    #                     out_spks[n_ind] = curr_time # Generate spike
    #                     active_neurons[n_ind] = 0 # Deactivate neuron
    #                     # NOTE : do not reset membrane potential because it can be used for decision-making
    #             # Update current timestamp only after generating spikes
    #             curr_time = time
    #         # Force to stop the simulation if an input spike occurs at max_time
    #         if time == limit_time: break
    #         # Record input spikes in a dense tensor (easier for the STDP update)
    #         # Dropout on the input is applied during the STDP update only
    #         # So the "dropped" input spikes still increase membrane potential
    #         if mask_in is None or mask_in[int(ind)] == 1: in_spks[int(ind)] = time
    #         # Increment membrane potentials of active neurons
    #         for n_ind in range(n_neurons):
    #             if active_neurons[n_ind] == 1:
    #                 mem_pots[n_ind] += weights[n_ind,int(ind)]
    for n_ind in range(n_neurons):
        if active_neurons[n_ind] == 0: continue
        
        # 각 뉴런에 도달하는 스파이크들의 [도착시간, 가중치] 리스트 생성
        arrival_times = []
        arrival_weights = []
        for ind, time in zip(indices, timestamps):
            if time >= limit_time: break
            t_arrival = time + delays[n_ind, int(ind)]
            arrival_times.append(t_arrival)
            arrival_weights.append(weights[n_ind, int(ind)])
        
        # 도착 시간 순으로 정렬 (Numba 호환 정렬)
        arrival_times = np.array(arrival_times)
        arrival_weights = np.array(arrival_weights)
        sort_idx = np.argsort(arrival_times)
        
        curr_mem = 0.0
        last_t = 0.0
        for idx in sort_idx:
            t_arr = arrival_times[idx]
            w = arrival_weights[idx]
            
            # Leak 적용
            if leak_tau is not None:
                tau = leak_tau[n_ind]
                if tau == 0:
                    print("Warning: leak_tau is set to 0, which may cause division by zero.")
                if tau > 0: # 0으로 나누기 방지
                    # t_arr와 last_t가 같은 경우(동시 도착) exp(0) = 1이 되므로 안전함
                    curr_mem *= np.exp(-(t_arr - last_t) / tau)
            
            curr_mem += w
            last_t = t_arr
            
            # 발화 체크
            if curr_mem > thresholds[n_ind]:
                out_spks[n_ind] = t_arr
                active_neurons[n_ind] = 0
                break
    return in_spks, out_spks, mem_pots



# Multiplicative STDP
@njit
def stdp_multiplicative(weights, in_spks, t_post, ap, am, error, w_min, w_max, beta):
    n_inputs = weights.shape[0]
    weights_out = weights.copy()
    for i in range(n_inputs):
        if in_spks[i] <= t_post:
            weights_out[i] += error * ap * math.exp(-beta * ((weights[i] - w_min) / (w_max-w_min)))
        else:
            weights_out[i] += error * am * math.exp(-beta * ((w_max - weights[i]) / (w_max-w_min)))
    return weights_out



# Additive STDP
@njit
def stdp_additive(weights, in_spks, t_post, ap, am, error, max_time):
    n_inputs = weights.shape[0]
    weights_out = weights.copy()
    for i in range(n_inputs):
        if in_spks[i] <= t_post:
            weights_out[i] += error * ap
        elif in_spks[i] < max_time + 1.0: # set to np.inf to consider all non-causal inputs
            weights_out[i] += error * am
    return weights_out



# Intra-class WTA
@njit
def class_inhibition(spks, pots, decision_map, max_time):
    out_spks_inhib = spks.copy()
    mem_pots_inhib = pots.copy()
    winners = []
    for c in np.arange(decision_map.n_classes, dtype=np.int32):
        n_idx = np.arange(c*decision_map.n_neurons_per_class, (c+1)*decision_map.n_neurons_per_class, dtype=np.int32)
        sorted_inds = spike_sort(pots[n_idx], spks[n_idx])
        winner = n_idx[sorted_inds[0]]
        losers = n_idx[sorted_inds[1:]]
        out_spks_inhib[losers] = max_time
        mem_pots_inhib[losers] = 0
        winners.append(winner)
    return out_spks_inhib, mem_pots_inhib, np.array(winners)

# SSTDP and S2-STDP weight update
@njit
def s2stdp(outputs, network_weights, y, decision_map, t_gap, class_inhib, use_time_ranges, max_time, ap, am, anti_ap, anti_am, stdp_func, stdp_args):
    n_layers = len(outputs)
    n_neurons_per_class = decision_map.n_neurons_per_class
    limit_time = max_time + 1.0
    # --- Compute the error for each layer --- #
    errors = []
    for layer_ind in range(n_layers-1,-1,-1): # Reversed loop

        # Extract the outputs of the forward pass 
        _, out_spks, mem_pots = outputs[layer_ind]
        
        # Init layer error
        n_neurons = out_spks.shape[0]
        error = np.zeros(n_neurons, dtype=np.float32)
        
        # Updated by Wonmo
        # Exclude non-active neurons from updating the error, spike time, and membrane potential
        # for n_ind in range(n_neurons):
        #     if decision_map.neuron_active[n_ind] == 0:
        #         error[n_ind] = 0
        #         out_spks[n_ind] = max_time + 1.0
        #         mem_pots[n_ind] = 0
                
        # Output layer
        if layer_ind == n_layers - 1:
            # Get neurons to update
            if class_inhib: # Intra-class WTA
                out_spks, mem_pots, to_update_neurons = class_inhibition(out_spks, mem_pots, decision_map, limit_time)
                n_target_neurons = 1
                n_updating_neurons = len(to_update_neurons)
            else: # No WTA
                n_target_neurons = n_neurons_per_class
                n_updating_neurons = n_neurons
                to_update_neurons = np.arange(n_neurons, dtype=np.int32)

            # Target and non-target desired timestamps based on the average firing time
            
            ntarget_t_gap = t_gap * (n_target_neurons/n_updating_neurons)
            target_t_gap = t_gap * ((n_updating_neurons-n_target_neurons)/n_updating_neurons)
            
            if np.all(out_spks == limit_time): t_base = limit_time - ntarget_t_gap
            else: t_base = np.minimum(np.mean(out_spks[out_spks < limit_time]), limit_time - ntarget_t_gap)
                      
            # Compute error for neurons to update
            for n_ind in to_update_neurons:
                
                # Updated by Wonmo
                # Skip non-active neurons from calculating error
                # if decision_map.neuron_active[n_ind] == 0: continue
                #######
                
                # Target neuron
                if decision_map.get_class(n_ind) == y and decision_map.is_target_neuron(n_ind):
                    # SSTDP training
                    if use_time_ranges: target = min(out_spks[n_ind], t_base - target_t_gap)
                    # S2-STDP training
                    else: target = t_base - target_t_gap
                # Non-target neuron
                else:
                    # SSTDP training
                    if use_time_ranges: target = max(out_spks[n_ind], t_base + ntarget_t_gap)
                    # S2-STDP training
                    else: target = t_base + ntarget_t_gap
                error[n_ind] = (out_spks[n_ind] - target) / limit_time

        # Hidden layer
        else:
            pass # NOT IMPLEMENTED

        # Keep track of the layer error
        errors.insert(0, error)

    # --- Compute the new weights --- #
    updated_weights = []
    # updated_delays = []
    
    for layer_ind in range(n_layers-1,-1,-1): # Reversed loop
        in_spks, out_spks, _ = outputs[layer_ind]
        weights = network_weights[layer_ind].copy()
        # delays = network_delays[layer_ind].copy()
        
        for n_ind,error in enumerate(errors[layer_ind]):
            
            # Updated by Wonmo
            # Skip non-active neurons from being updated
            # if decision_map.neuron_active[n_ind] == 0: continue
            #######
            
            # No update
            if error == 0: continue
            # Select the learning rate to apply STDP or anti-STDP
            if np.sign(error) < 0: lrp, lrm = anti_ap, anti_am
            else: lrp, lrm = ap, am
            # Compute weight update
            weights[n_ind] = stdp_func(weights[n_ind], in_spks, out_spks[n_ind], lrp, lrm, abs(error), *stdp_args)
            
            # t_target = out_spks[n_ind] - (error * limit_time)
            
            # delays[n_ind] = delay_learning_rule(
            #     delays[n_ind], in_spks, out_spks[n_ind], t_target, lr_delay, d_min, d_max
            # )
            
        updated_weights.insert(0, weights)
        # updated_delays.insert(0, delays)
        
    return updated_weights



# S4NN backward pass
# Almost similar to sstdp code, but in another function for the sake of clarity
@njit
def s4nn(outputs, network_weights, y, decision_map, t_gap, class_inhib, use_time_ranges, max_time, lr):
    n_layers = len(outputs)

    # --- Compute the error for each layer --- #
    errors = []
    for layer_ind in range(n_layers-1,-1,-1): # Reversed loop

        # Extract the outputs of the forward pass 
        _, out_spks, mem_pots = outputs[layer_ind]
        
        # Init layer error
        n_neurons = out_spks.shape[0]
        error = np.zeros(n_neurons, dtype=np.float32)
        
        # Output layer
        if layer_ind == n_layers - 1:
            
            # Get neurons to update
            if class_inhib: # Intra-class WTA
                out_spks, mem_pots, to_update_neurons = class_inhibition(out_spks, mem_pots, decision_map, limit_time)
            else: # No WTA
                to_update_neurons = np.arange(n_neurons)
            
            # Target and non-target desired timestamps based on the min firing time   
            t_base = out_spks.min()
            if t_base + t_gap > limit_time: t_base = limit_time - t_gap
                      
            # Compute error for neurons to update
            for n_ind in to_update_neurons:
                # Target neuron
                if decision_map.get_class(n_ind) == y and decision_map.is_target_neuron(n_ind):
                    # Original S4NN
                    if use_time_ranges: target = min(out_spks[n_ind], t_base)
                    # Adapted version
                    else: target = t_base
                # Non-target neuron
                else:
                    # Original S4NN
                    if use_time_ranges: target = max(out_spks[n_ind], t_base + t_gap)
                    # Adapted version
                    else: target = t_base + t_gap
                error[n_ind] = (out_spks[n_ind] - target) / limit_time

        # Hidden layer
        else:
            pass # NOT IMPLEMENTED
        
        # Gradient normalization
        norm = np.linalg.norm(error)
        if (norm != 0): error = error / norm

        # Keep track of the layer error
        errors.insert(0, error)

    # --- Compute the new weights --- #
    updated_weights = []
    for layer_ind in range(n_layers-1,-1,-1): # Reversed loop
        in_spks, out_spks, _ = outputs[layer_ind]
        weights = network_weights[layer_ind].copy()
        for n_ind,error in enumerate(errors[layer_ind]):
            # No update
            if error == 0: continue
            # Compute weight update
            for i in range(weights[n_ind].shape[0]):
                if in_spks[i] <= out_spks[n_ind]:
                    weights[n_ind,i] += error * lr
        updated_weights.insert(0, weights)
        
    return updated_weights