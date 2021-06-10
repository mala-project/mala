![image](./docs/source/img/logos/mala_horizontal.png)

# MALA

[![image](https://github.com/mala-project/mala/actions/workflows/gh-pages.yml/badge.svg)](https://mala-project.github.io/mala/)

[![image](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


MALA (Materials Learning Algorithms) is a data-driven framework to generate surrogate models of density functional theory calculations based on machine learning. Its purpose is to enable multiscale modeling by bypassing computationally expensive steps in state-of-the-art density functional simulations.

MALA is designed as a modular and open-source python package. It enables users to perform the entire modeling toolchain using only a few lines of code. MALA is jointly developed by the Sandia National Laboratories (SNL) and the Center for Advanced Systems Understanding (CASUS). See Contributing.md for contributing code to the repository.

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

## Institutions
### Founding Institutions

- Sandia National Laboratories (SNL), USA.
- Center for Advanced Systems Understanding (CASUS), Germany.

## Developers
### Scientific Supervision
- Attila Cangi (CASUS)
- Siva Rajamanickam (SNL)

### Core Developers
- Lenz Fiedler (CASUS)
- Austin Ellis (ORNL*)
- Normand Modine (SNL)
- Steve Schmerler (CASUS)
- Daniel Kotik (CASUS)
- Gabriel Popoola (SNL)
- Aidan Thompson (SNL)
- Adam Stephens (SNL)

\* Work done as part of postdoctoral research at Sandia National Laboratories


## Citing MALA

If you publish work which uses or mentions MALA, please cite the following paper:

J. A. Ellis, G. A. Popoola, L. Fiedler, N. A. Modine, J. A. Stephens, A. P. Thompson, 
A. Cangi, S. Rajamanickam (2020). Accelerating Finite-temperature
Kohn-Sham Density Functional Theory with Deep Neural Networks.
[arXiv:2010.04905](https://arxiv.org/abs/2010.04905).
