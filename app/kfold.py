import sys
sys.path.insert(0, "/data/0312/SpikeNN/")
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
    parser.add_argument("--K", type=int, default=10, help="Number of folds")
    parser.add_argument("--n_proc", type=int, default=10, help="Number of processes")
    parser.add_argument("--run_name", type=str, help="name of wandb recording")
    args = parser.parse_args()

    # Read the JSON config
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # updated by Wonmo
    # wandb does not work well with multiprocessing, so changed to execute K-fold sequentially as a temporary solution
    for i in range(int(args.K)):
        print('Running fold', i)
        main(f"{args.input_dir}/{i}", config, f"{args.output_dir}/{i}/", i, args.run_name + f"_fold_{i}" if args.run_name else None)
    # # Create a pool of workers for parallel processing
    # pool = multiprocessing.Pool(processes=int(args.n_proc))

    # results = pool.starmap_async(main, [(f"{args.input_dir}/{i}", config, f"{args.output_dir}/{i}/", i) for i in range(int(args.K))])
    
    # results.get()

    # # End the pool
    # pool.close()
    # pool.join()