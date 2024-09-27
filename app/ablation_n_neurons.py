import os
import json
import argparse
import multiprocessing
from run import main


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input data directory")
    parser.add_argument("output_dir", type=str, help="Output data directory")
    parser.add_argument("config_path", type=str, help="JSON config path")
    parser.add_argument("--seed", nargs="?", type=int, default=0, help="Random seed")
    parser.add_argument("--K", type=int, default=10, help="Number of folds")
    parser.add_argument("--n_proc", type=int, default=10, help="Number of processes")
    parser.add_argument('--is_rstdp', action='store_true', help='Run for R-STDP')
    args = parser.parse_args()

    # Read the JSON config
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    for n_neurons in [30, 50, 70, 100, 150]:
        
        if os.path.isdir(f"{args.output_dir}/{str(n_neurons)}/"): continue 
                
        # To ensure it is OK with higher number of neurons
        config["trainer"]["epochs"] = 200

        config["network"][0]["n_neurons"] = n_neurons

        # Create a pool of workers for parallel processing (K-fold)
        pool = multiprocessing.Pool(processes=int(args.n_proc))

        results = pool.starmap_async(main, [(f"{args.input_dir}/{i}/", config, f"{args.output_dir}/{str(n_neurons)}/{i}/", i) for i in range(int(args.K))])
        
        results.wait()

        # End the pool
        pool.close()
        pool.join()
