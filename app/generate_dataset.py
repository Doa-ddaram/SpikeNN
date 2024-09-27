import os
import argparse
import numpy as np
from spikenn.dataset import SpikingDataset


def generate_dataset(data_path, label_path, output_path, rm_input=True):
    # Load the input
    X = np.load(data_path)
    y = np.load(label_path)
    assert X.shape[0] == len(y)

    # Create and save the SpikingDataset object
    SpikingDataset.from_numpy(X, y, max_time=1).save(output_path)

    print(f"Successfully converted ({data_path} ; {label_path}) to {output_path}")
    
    # Remove the old file if specified
    if rm_input:
        os.remove(data_path)
        os.remove(label_path)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_path", help="Input data path")
    parser.add_argument("input_label_path", help="Input label path")
    parser.add_argument("output_path", help="Output path")
    parser.add_argument("rm_input", nargs="?", type=bool, default=True, help="Remove the input files")
    args = parser.parse_args()
    
    generate_dataset(args.input_data_path, args.input_label_path, args.output_path, args.rm_input)
