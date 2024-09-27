# SpikeNN

Spiking Neural Network (SNN) framework for classification, featuring fully-connected architectures, first-spike coding, single-spike IF/LIF neurons, floating-point spike timestamps (event-driven), and STDP-based supervised learning rules.

This is the code for the paper *Neuronal Competition Groups with Supervised STDP for Spike-Based Classification*, published at NeurIPS 2024 (see [official repo](https://gitlab.univ-lille.fr/fox/snn-ncg)).


## Getting started

The code is written with Python3 (3.8.10) and **runs exclusively on CPUs**.

### Requirements

- numpy (1.23.4)
- numba (0.56.4)
- tqdm (4.64.1)
- setuptools (45.2.0)

### Install
  
The `spikeNN` package and its dependencies can be installed with:
```
python3 -m pip install -e .
```

In case you get any errors with a dependency, please try to use the same version as ours.


## Usage

Applications are located in the `app/` folder.   
Configuration files are located in the `config/` folder.  
The `spikeNN` package provides all the core classes and functions (see `spikenn/`).  
The `Readout` class (`app/readout.py`) defines a flexible SNN model for classification using the `spikeNN` package.  

### Input data

This framework is designed to process input data wrapped into `SpikingDataset` objects, where each sample is represented by an ordered sparse list of spikes.
A spike is a tuple `<i, t>` (i: 1D-index of the input neuron, t: firing timestamp). 

The easiest way to convert a dense numpy dataset (here, training set) into a `SpikingDataset` is as follows:
```
python3 app/generate_dataset.py /input/X_train.npy /input/y_train.npy /output/trainset.npy 
```
- `X_train.npy`: numpy array of shape *(N_samples, D1, ..., DN)*, where each value is a spike timestamp.
- `y_train.npy`: numpy array of shape *(N_samples,)*, where each value is a class label.


### Run 

To start a single run:
```
python3 app/run.py /input/data/dir/ /output/data/dir/ /config/file [--seed 0]
```

To start a K-fold run:
```
python3 app/kfold.py /input/data/dir/ /output/data/dir/ /config/file [--K 10] [--n_proc 10]
```
When the run is done, you can use `python3 app/get_kfold_score.py /output/data/dir/` to compute the mean accuracy.

To start a gridsearch run:
```
python3 app/gridsearch.py /input/data/dir/ /output/data/dir/ /config/file [--n_proc MAX] [--resume]
```
When the run is done, you can use `python3 app/get_gs_best.py /output/data/dir/` to show the best gridsearch runs.

**Notes:**
- Input directory must contain `SpikingDataset` files with the following name convention: `trainset.npy`, `valset.npy` (optional), `testset.npy` (optional).
- Configuration files use the JSON format and follow a specific syntax (see examples in `config/`).
- Use `python3 <app_file>.py -h` for additional information regarding arguments.


## Reproduce the main results

All our quantitative results are computed with K-fold runs.  
All the configuration files employed in our experiments are located in the `config/` folder.  
Gridsearch-optimized configurations are placed in the `config/<FT-EXTRACTOR>/<DATASET>/` folders and gridsearch configurations in the `gs/` sub-folders.  

### Unsupervised feature extraction

In the paper, datasets are preprocessed with Hebbian-based unsupervised feature extractors before classification:
- STDP-CSNN: a single-layer spiking CNN trained with STDP ([original repository](https://gitlab.univ-lille.fr/bioinsp/falez-csnn-simulator)).
- SoftHebb-CNN: a three-layer non-spiking CNN trained with SoftHebb ([original repository](https://github.com/NeuromorphicComputing/SoftHebb)).

The folder `ft-extract/` provides the scripts and documentation for obtaining and preprocessing each dataset with these feature extractors, as well as converting them to `SpikingDataset` objects.

We also provide some features extracted with STDP-CSNN to facilitate quick experimentation: *To be updated*.

### 5.2 Accuracy comparison

The following command can be used to reproduce our scores from Table 1:  
```
python3 app/kfold.py ft-extract/extracted/<FT-EXTRACTOR>/<DATASET>/kfold/ logs/<FT-EXTRACTOR>/<DATASET>/kfold/<CONFIG>/ config/<FT-EXTRACTOR>/<DATASET>/<CONFIG>.json
```
- Replace `<DATASET>` by: `MNIST`, `F-MNIST`, `CIFAR10`, `CIFAR100`.  
- Replace `<FT-EXTRACTOR>` by: `STDP-CSNN`, `SoftHebb-CNN`.  
- Replace `<CONFIG>` by: `s2stdp+ncg`, `sstdp+ncg`, `s2stdp`, `sstdp`, `rstdp-5n`, `rstdp-20n`.

Output will be logged in `.txt` files at `logs/<FT-EXTRACTOR>/<DATASET>/kfold/<CONFIG>/`.

### 5.3 Ablation study

Here are the names of the configuration files used in for the ablation study on S2-STDP+NCG:
- *M-1*: `s2stdp.json`
- *M-5*: `s2stdp+ncg-no-cr-no-nt.json`
- *M-5+CR-1*: `s2stdp+ncg-cr-1-thr-no-nt.json`
- *M-5+CR-2*: `s2stdp+ncg-no-nt.json`
- *M-5+CR-1+L*: `s2stdp+ncg-cr-1-thr.json`
- *M-5+CR-2+L*: `s2stdp+ncg.json`
- *M-5+Drop+L*: `s2stdp+ncg-cr-drop.json`

### 5.4 Impact of competition regulation

In `app/readout.py`, the variable `self.save_stats` must be set to `True` to save the training logs used to generate the figures.


## Limitations

- Limited to first-spike coding (TTFS) with single-spike IF/LIF neurons
- No support for other architectures than fully-connected
- Multi-layer training not implemented
- Long computation times (CPU-based)


## Known bugs

- Results may vary slightly, though not significantly, depending on the version of Numba installed.

**Note:** if you encounter any issues with the code, please report them by creating an issue.


## Acknowledgments

- [Chaire Luxant-ANVI](https://chaire-luxant-anvi.univ-lille.fr/) (Métropole de Lille)
- UAR 3380 - IRCICA - Institut de recherche sur les composants logiciels et matériels pour l'information et la communication avancée, Lille, F-59000, France
- Univ. Lille, CNRS, Centrale Lille, MR 9189 - CRIStAL - Centre de Recherche en Informatique, Signal et Automatique de Lille, Lille, F-59000, France