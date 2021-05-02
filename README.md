![image](./docs/source/img/logos/mala_horizontal.png)

# MALA

[![image](https://github.com/mala-project/mala/actions/workflows/gh-pages.yml/badge.svg)](https://mala-project.github.io/mala/)

[![image](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


MALA (Materials Learning Algorithms) is a data-driven framework to generate surrogate models of density functional theory calculations based on machine learning. Its purpose is to enable multiscale modeling by bypassing computationally expensive steps in state-of-the-art density functional simulations.

MALA is designed as a modular and open-source python package. It enables users to perform the entire modeling toolchain using only a few lines of code. MALA is jointly developed by the Sandia National Laboratories (SNL), the Oak Ridge National Laboratory (ORNL), and the Center for Advanced Systems Understanding (CASUS). 
This repository is structured as follows:
```
├── examples : contains useful examples to get you started with the package
├── install : contains scripts for setting up this package on your machine
├── mala : the source code itself
├── test : test scripts used during development, will hold tests for CI in the future
└── docs : Sphinx documentation folder
```

## Installation

Please refer to [Installation of MALA](docs/source/install/README.md).

## Running

You can familiarize yourself with the usage of this package by running
the examples in the `example/` folder.

## Developers
### Scientific Supervision
- Attila Cangi
- Siva Rajamanickam

### Core Developers
- Lenz Fiedler
- Steve Schmerler
- Daniel Kotik

### Contributors
- Omar Faruk
- Parvez Mohammed
- Sneha Verma
- Somashekhar Kulkarni

### Former Members
- Nils Hoffmann


## Citing MALA

If you publish work which uses or mentions MALA, please cite the following paper:

J. A. Ellis, A. Cangi,  N. A. Modine, J. A. Stephens, A. P. Thompson,
S. Rajamanickam (2020). Accelerating Finite-temperature
Kohn-Sham Density Functional Theory with Deep Neural Networks.
[arXiv:2010.04905](https://arxiv.org/abs/2010.04905).
