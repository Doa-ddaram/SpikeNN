from setuptools import setup, find_packages

setup(
    name='spikeNN',
    version='1.0.0',
    description='Spiking neural network framework for classification, featuring fully-connected architectures, first-spike coding, single-spike IF/LIF neurons, floating spike timestamps (event-driven), and STDP-based supervised learning rules.', 
    keywords='SNN, TTFS coding, STDP, WTA competition, supervised local learning, classification',
    packages=find_packages(),
    install_requires=['numpy', 'numba', 'tqdm'],
    python_requires='>=3',
)
