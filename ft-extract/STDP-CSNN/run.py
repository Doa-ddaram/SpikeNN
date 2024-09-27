import os
import shutil
import argparse
import subprocess
import numpy as np

from spikenn.dataset import SpikingDataset

"""
Single-file script to train a single-layer CSNN using STDP and save the output spike features.

Code is adapted from https://gitlab.univ-lille.fr/fox/snn-pcn/
"""

# Get the root path of the directory
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

# Return the absolute path of a relative path
def absolute_path(path, isdir=True):
    return os.path.abspath(path) + ("/" if isdir else "")


# Run the CSNN
def run(input_dir, config_path, output_dir, seed):
    # Create output directory
    os.makedirs(output_dir)
    # Train CSNN on training data
    subprocess.check_call([f'{ROOT_PATH}/csnn/build/csnn_train', f'{input_dir}/X_train.bin', f'{input_dir}/y_train.bin', config_path, output_dir, str(seed)])
    # Convert CSNN feature maps to correct format
    SpikingDataset.from_numpy(np.load(f'{output_dir}/X_csnn_train.npy'), np.load(f'{output_dir}/y_train.npy'), max_time=1).save(f'{output_dir}/trainset.npy')
    os.remove(f'{output_dir}/X_csnn_train.npy')
    os.remove(f'{output_dir}/y_train.npy')
    # Test CSNN on validation data
    if os.path.exists(f'{input_dir}/X_val.bin'):
        subprocess.check_call([f'{ROOT_PATH}/csnn/build/csnn_test', f'{input_dir}/X_val.bin', f'{input_dir}/y_val.bin', f'{output_dir}/model/', output_dir])
        SpikingDataset.from_numpy(np.load(f'{output_dir}/X_csnn_test.npy'), np.load(f'{output_dir}/y_test.npy'), max_time=1).save(f'{output_dir}/valset.npy')
        os.remove(f'{output_dir}/X_csnn_test.npy')
        os.remove(f'{output_dir}/y_test.npy')
    # Test CSNN on test data
    if os.path.exists(f'{input_dir}/X_test.bin'):
        subprocess.check_call([f'{ROOT_PATH}/csnn/build/csnn_test', f'{input_dir}/X_test.bin', f'{input_dir}/y_test.bin', f'{output_dir}/model/', output_dir])
        SpikingDataset.from_numpy(np.load(f'{output_dir}/X_csnn_test.npy'), np.load(f'{output_dir}/y_test.npy'), max_time=1).save(f'{output_dir}/testset.npy')
        os.remove(f'{output_dir}/X_csnn_test.npy')
        os.remove(f'{output_dir}/y_test.npy')
    # Remove CSNN files
    os.remove(f'{output_dir}/log_train.txt')
    shutil.rmtree(f'{output_dir}/model')


# --------------------------------------------- #
# --------------------------------------------- #


if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input data directory")
    parser.add_argument("output_dir", type=str, help="Output data directory")
    parser.add_argument("config_path", type=str, help="Configuration path of the CSNN")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Absolute paths
    input_dir = absolute_path(args.input_dir)
    output_dir = absolute_path(args.output_dir)
    config_path = absolute_path(args.config_path, isdir=False)

    # Assert that the output directory does not exist
    if os.path.isdir(output_dir): raise RuntimeError("Output directory already exists. Aborting...")

    try:
        run(input_dir, config_path, output_dir, args.seed)
    except Exception as e:
        print("RUNNING ERROR:\n", e)