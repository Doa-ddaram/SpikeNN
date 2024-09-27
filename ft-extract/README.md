# Hebbian-based unsupervised feature extractors

This folder provides scripts to extract, in an unsupervised fashion, spike features from image recognition datasets using [STDP-CSNN](https://gitlab.univ-lille.fr/bioinsp/falez-csnn-simulator) and [SoftHebb-CNN](https://github.com/NeuromorphicComputing/SoftHebb).

Following first-spike coding, each output feature is a single floating-point spike timestamp in `[0, 1]`.


## Prepare the datasets

### Requirements

#### Python3
- numpy (1.23.4)
- scikit_learn (1.1.3)
- requests (2.22.0)

### Run

In `dataset/`, we provide the scripts: `get_mnist.py`, `get_fmnist.py`, `get_cifar10.py`, `get_cifar100.py`.

These scripts download and format the datasets, then generate the gridsearch (`gs/`) and K-fold (`kfold/`) splits used in our experiments.


## STDP-CSNN

The CSNN simulator (in `STDP-CSNN/csnn/`) is written with C++ and is a fork of this [repository](https://gitlab.univ-lille.fr/fox/snn-pcn).   
This simulator has been tested only with Ubuntu 20.04 LTS. 

### Requirements

#### C++
- C++ compiler (version >= 14)
- Cmake (version >= 3.1)
- BLAS
- LAPACKE

#### Python3
- numpy (1.23.4)
- spikenn (1.0.0)

### Install

To install the requirements with an apt-based distribution:
```
sudo apt install --yes gcc g++ make cmake libatlas-base-dev libblas-dev libopenblas-dev liblapack-dev liblapacke-dev
```
To compile the code:
```
cd STDP-CSNN/csnn/ && mkdir -p build/ && cd build/ && cmake ../ -DCMAKE_BUILD_TYPE=Release && make 
```

### Run
From `STDP-CSNN/`,

To generate the extracted features for the gridsearch experiments:
```
python3 run.py ../dataset/<DATASET>/gs/ ../extracted/STDP-CSNN/<DATASET>/gs/ csnn/config/<DATASET>.json
```

To generate the extracted features for the K-fold experiments:
```
for i in {0..9}; do python3 run.py ../dataset/<DATASET>/kfold/$i/ ../extracted/STDP-CSNN/<DATASET>/kfold/$i/ csnn/config/<DATASET>.json --seed=$i; done
```

Replace `<DATASET>` by: `MNIST`, `F-MNIST`, `CIFAR10`, `CIFAR100`.  


## SoftHebb-CNN

The provided scripts (in `SoftHebb-CNN/`) are written with Python3 and are adapted from `demo.py` in the [SoftHebb repository](https://github.com/NeuromorphicComputing/SoftHebb).

### Requirements

#### Python3
- numpy (1.23.4)
- torch (2.0.1)
- torchvision (0.15.2)
- spikenn (1.0.0)


### Run

From `SoftHebb-CNN/`,

To generate the extracted features for the gridsearch experiments:
```
python3 <run_script> ../dataset/<DATASET>/gs/ ../extracted/SoftHebb-CNN/<DATASET>/gs/
```

To generate the extracted features for the K-fold experiments:
```
for i in {0..9}; do python3 <run_script> ../dataset/<DATASET>/kfold/$i/ ../extracted/SoftHebb-CNN/<DATASET>/kfold/$i/; done
```

Replace:
- `<run_script>` by `run_mnist.py` (MNIST/Fashion-MNIST), `run_cifar.py` (CIFAR-10/CIFAR-100).
- `<DATASET>` by: `MNIST`, `F-MNIST`, `CIFAR10`, `CIFAR100`.  


## Known bugs
- MNIST can not be downloaded anymore at http://yann.lecun.com/ (403 Forbidden).