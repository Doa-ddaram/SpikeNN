import os
import json
import shutil
import argparse
import itertools
import multiprocessing
from run import main


# Take a JSON config file, with <type> for constant attributes and List<type> for variable attributes, 
# and return a list of config dicts with all possible solutions
# NOTE: Not very clean :)
def create_configs_from_subdict(subdict):
    # Get the names of the attributes containing arrays
    array_attribute_names = [k for k, v in subdict.items() if isinstance(v, list)]
    # Create a list of all possible combinations of values for the attributes containing arrays
    combinations = [{k: v[i] for i, k in enumerate(array_attribute_names)}
                        for v in itertools.product(*(subdict[k] 
                            for k in array_attribute_names))]
    # Create the configs with the different combinations
    all_configs = [{k: v if k not in array_attribute_names else combination[k] 
                        for k,v in subdict.items()} 
                            for combination in combinations]
    return all_configs

def create_configs_from_gridsearch_dict(gs_dict):
    # Optimizer configs
    optimizer_configs = create_configs_from_subdict(gs_dict["optimizer"])

    # Regularizer configs
    regularizer_configs = None
    if "regularizer" in gs_dict:
        regularizer_configs = create_configs_from_subdict(gs_dict["regularizer"])

    # Trainer configs
    trainer_configs = create_configs_from_subdict(gs_dict["trainer"])
    
    # Network configs
    network_configs = []
    for layer in gs_dict["network"]: network_configs.append(create_configs_from_subdict(layer))

    # Create all combinations
    all_configs = []
    for layers in itertools.product(*network_configs):
        network = [layer for layer in layers]
        if regularizer_configs is not None:
            for regul in regularizer_configs:
                for optim in optimizer_configs:            
                    for trainer in trainer_configs:
                        d = {"network": network, "optimizer": optim, "trainer": trainer, "regularizer": regul}
                        all_configs.append(d)
        else:
            for optim in optimizer_configs:            
                for trainer in trainer_configs:
                    d = {"network": network, "optimizer": optim, "trainer": trainer}
                    all_configs.append(d)
    
    return all_configs


def dispatch(input_dir, config, output_dir, seed, resume=False):
    # Resume mode
    if resume:
        try:
            if os.path.exists(output_dir) and os.path.isdir(output_dir):
                output_file = output_dir + '/readout_log.txt'
                if os.path.exists(output_file) and os.path.isfile(output_file):
                    with open(output_file, 'r') as file:
                        content = file.read()
                        if "Early stopping triggered" in content:
                            # Run done
                            return
                        else:
                            # Redo the run
                            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"An error occurred: {e}")
    main(input_dir, config, output_dir, seed)
    
    
if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input data directory")
    parser.add_argument("output_dir", type=str, help="Output data directory")
    parser.add_argument("config_path", type=str, help="JSON config path")
    parser.add_argument("--n_proc", type=int, default=-1, help="Number of processes")
    parser.add_argument('--resume', action='store_true', help='Resume an unfinished run')
    args = parser.parse_args()

    # Create a pool of workers for parallel processing
    if args.n_proc == -1: n_proc = multiprocessing.cpu_count() - 1
    else: n_proc = int(args.n_proc)
    pool = multiprocessing.Pool(processes=n_proc)

    # Create configs
    with open(args.config_path, "r") as file:
        # Create all combinations possible
        readout_configs = create_configs_from_gridsearch_dict(json.load(file))        
        
    # Run exps
    results = pool.starmap_async(dispatch, [(args.input_dir, config, f"{args.output_dir}/{run_ind}/", 0, args.resume)
                                                    for run_ind,config in enumerate(readout_configs)])
    results.wait() # Wait for all gridsearch steps to be done

    # End the pool
    pool.close()
    pool.join()