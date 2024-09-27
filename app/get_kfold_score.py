import os
import argparse
import numpy as np


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("kfold_dir", type=str, help="Kfold data directory")
    args = parser.parse_args()

    # Count the number of folds
    kfold_dir_runs = []
    for item in os.listdir(args.kfold_dir):
        full_path = os.path.join(args.kfold_dir, item)
        if os.path.isdir(full_path): kfold_dir_runs.append(full_path)
    print(f"Reading {len(kfold_dir_runs)}-folds...")

    train_accs = []
    train_epochs = []
    val_accs = []
    test_accs = []
    for run_dir in kfold_dir_runs:
        with open(run_dir + "/readout_log.txt", 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "training set" in line:
                    train_acc = float(line.replace(" ", "").split(':')[1])
                    train_epoch = int(line.split(':')[0].split(" ")[-1])
                if "validation set" in line:
                    val_acc = float(line.replace(" ", "").split(':')[1])
                if "test set" in line:
                    test_acc = float(line.replace(" ", "").split(':')[1])
            train_accs.append(train_acc)
            train_epochs.append(train_epoch)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
    
    print(f"### TRAIN ###")
    print(f"Accuracies: {train_accs}")
    print(f"Average number of epochs: {np.mean(train_epochs)}")
    print(f"{str(round(np.mean(train_accs)*100,2))} +- {str(round(np.std(train_accs)*100,2))}")
    print()
    print(f"### VAL ###")
    print(f"Accuracies: {val_accs}")
    print(f"{str(round(np.mean(val_accs)*100,2))} +- {str(round(np.std(val_accs)*100,2))}")
    print()
    print(f"### TEST ###")
    print(f"Accuracies: {test_accs}")
    print(f"{str(round(np.mean(test_accs)*100,2))} +- {str(round(np.std(test_accs)*100,2))}")
