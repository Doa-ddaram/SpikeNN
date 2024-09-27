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
    parser.add_argument("--thr_lr", nargs='+', type=str, default=[15, 10, 5, 1, 0.5, 0.1, 0.05, 0.01], help="List of threshold learning rate")
    parser.add_argument("--thr_anneal", nargs='+', type=str, default=[1, 0.9, 0.5, 0.1], help="List of threshold annealing")
    args = parser.parse_args()

    # Read the JSON config
    with open(args.config_path, "r") as f:
        config = json.load(f)

    for n_neurons in [50]: # Fixed for this work
        
        for thr_lr in args.thr_lr:
            
            for thr_anneal in args.thr_anneal:
            
                run_dir = f"{args.output_dir}/{str(n_neurons)}/{str(thr_lr)}/{str(thr_anneal)}"
                
                if os.path.isdir(run_dir): continue                
            
                config["network"][0]["n_neurons"] = int(n_neurons)
                config["regularizer"]["thr_lr"] = float(thr_lr)
                config["regularizer"]["thr_anneal"] = float(thr_anneal)
                
                # Create a pool of workers for parallel processing (K-fold)
                pool = multiprocessing.Pool(processes=int(args.n_proc))

                results = pool.starmap_async(main, [(f"{args.input_dir}/{i}/", config, f"{run_dir}/{i}/", i) for i in range(int(args.K))])
                
                results.wait()

                # End the pool
                pool.close()
                pool.join()
