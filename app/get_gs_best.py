import os
import argparse
import numpy as np


def get_run_dirs(path):
    dirs = []
    # loop through all directories in the given path
    for dirpath, dirnames, _ in os.walk(path):
        for dirname in dirnames: 
            dirs.append(os.path.join(dirpath, dirname))  
    return dirs


def get_readout_validation_accs(path):
    val_accs = []
    test_accs = []
    log_files = os.listdir(path)
    readout_log_file = None
    for f in log_files:
        if f.startswith("readout_log"):
            readout_log_file = path + "/" + f
    if readout_log_file is not None:
        with open(readout_log_file, "r") as file:
            for line in file:
                if "Accuracy on validation set" in line:
                    accuracy = float(line.split(':')[1])
                    val_accs.append(accuracy)
                elif "Accuracy on test set" in line:
                    accuracy = float(line.split(':')[1])
                    test_accs.append(accuracy)
    if len(val_accs) == 0 and len(test_accs) > 0: return test_accs
    else: return val_accs



if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Gridsearch log path")
    parser.add_argument("--N", default=10, help="Number of configs to show")
    args = parser.parse_args()

    exp_dirs = get_run_dirs(args.path)

    print(f"Reading {len(exp_dirs)} experiments...")

    val_accs = []
    epochs = []
    acc_stds = []
    for exp_dir in exp_dirs:
        accs = get_readout_validation_accs(exp_dir)
        if len(accs) > 0:
            acc, epoch = accs[-1], len(accs)-1
            acc_std = np.std(accs[-5:])
        else: acc, epoch, acc_std = 0, 0, -1
        val_accs.append(acc)
        epochs.append(epoch)
        acc_stds.append(acc_std)
    val_accs = np.array(val_accs)
    epochs = np.array(epochs)
    acc_stds = np.array(acc_stds)

    ordered_exps = val_accs.argsort()[::-1]
    val_accs = val_accs[ordered_exps]
    epochs = epochs[ordered_exps]
    acc_stds = acc_stds[ordered_exps]

    for i in range(int(args.N)):
        print(f"Top #{i+1} config is {exp_dirs[ordered_exps[i]]} with a validation accuracy of {val_accs[i]} (epoch:{epochs[i]} ; std:{acc_stds[i]}).")