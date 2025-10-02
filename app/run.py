import os
import json
import argparse
import numpy as np
from readout import Readout
from spikenn.dataset import SpikingDataset


def main(dataset_path, config, output_dir, seed=0):
    # Seeding
    np.random.seed(int(seed))

    # Load dataset
    # NOTE: data must be stored to the SpikingDataset format
    trainset = SpikingDataset.from_file(f"{dataset_path}/trainset.npy")
    valset = SpikingDataset.from_file(f"{dataset_path}/valset.npy") if os.path.exists(f"{dataset_path}/valset.npy") else None
    testset = SpikingDataset.from_file(f"{dataset_path}/testset.npy") if os.path.exists(f"{dataset_path}/testset.npy") else None
    
    # Get the number of classes
    n_classes = len(np.unique(trainset.labels))

    # Assert data is not empty
    empty_dataset = np.any([sample.size == 0 for sample,_ in trainset])
    if empty_dataset: raise RuntimeError("Some input samples do not contain any spike.")

    # Define the model
    model = Readout.init_from_dict(config, trainset.shape[1], n_classes, output_dir, trainset.max_time, neurons_per_class=[3]*10)
    model.logger.log(config)

    # Train the model
    model.fit(train_dataset=trainset, val_dataset=valset, test_dataset=testset)
    
    # Close the logger
    model.logger.stop()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input data directory")
    parser.add_argument("output_dir", type=str, help="Output data directory")
    parser.add_argument("config_path", type=str, help="JSON config path")
    parser.add_argument("--seed", nargs="?", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Read the JSON config
    with open(args.config_path, "r") as f:
        config = json.load(f)

    main(args.input_dir, config, args.output_dir, args.seed)
