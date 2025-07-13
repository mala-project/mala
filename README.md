![image](./docs/source/img/logos/mala_horizontal.png)

# MALA

[![CPU](https://github.com/mala-project/mala/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/mala-project/mala/actions/workflows/cpu-tests.yml)
[![image](https://github.com/mala-project/mala/actions/workflows/gh-pages.yml/badge.svg)](https://mala-project.github.io/mala/)
[![image](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5557255.svg)](https://doi.org/10.5281/zenodo.5557255)


MALA (Materials Learning Algorithms) is a data-driven framework to generate surrogate models of density functional theory calculations based on machine learning. Its purpose is to enable multiscale modeling by bypassing computationally expensive steps in state-of-the-art density functional simulations.

MALA is designed as a modular and open-source python package. It enables users to perform the entire modeling toolchain using only a few lines of code. MALA is jointly developed by the Sandia National Laboratories (SNL) and the Center for Advanced Systems Understanding (CASUS). See [Contributing](docs/source/CONTRIBUTE.md) for contributing code to the repository.

This repository is structured as follows:
```
├── examples : contains useful examples to get you started with the package
├── install : contains scripts for setting up this package on your machine
├── mala : the source code itself
├── test : test scripts used during development, will hold tests for CI in the future
└── docs : Sphinx documentation folder
```

## Installation

> **WARNING**: Even if you install MALA via PyPI, please consult the full installation instructions afterwards. External modules (like the QuantumESPRESSO bindings) are not distributed via PyPI!

Please refer to [Installation of MALA](docs/source/install/installing_mala.rst).

## Running

You can familiarize yourself with the usage of this package by running
the examples in the `example/` folder.

## Contributors

MALA is jointly maintained by 

- [Sandia National Laboratories](https://www.sandia.gov/) (SNL), USA.
    - Scientific supervisor: Sivasankaran Rajamanickam, code maintenance: 
Jon Vogel
- [Center for Advanced Systems Understanding](https://www.casus.science/) (CASUS), Germany.
    - Scientific supervisor: Attila Cangi, code maintenance: Lenz Fiedler

A full list of contributors can be found [here](docs/source/CONTRIBUTE.md).

## Citing MALA

If you publish work which uses or mentions MALA, please cite this repository and the following papers:

- A. Cangi, L. Fiedler, B. Brzoza, K. Shah, T. J. Callow, D. Kotik, S. Schmerler, M. C. Barry, J. M. Goff, A. Rohskopf, D. J. Vogel, N. Modine, A. P. Thompson, S. Rajamanickam,
Materials Learning Algorithms (MALA): Scalable Machine Learning for Electronic Structure Calculations in Large-Scale Atomistic Simulations,
[Comp. Phys. Commun. 314, 109654 (2025)](https://doi.org/10.1016/j.cpc.2025.109654).

- J. A. Ellis, L. Fiedler, G. A. Popoola, N. A. Modine, J. A. Stephens, A. P. Thompson,
A. Cangi, S. Rajamanickam, Accelerating Finite-Temperature Kohn-Sham Density Functional Theory with Deep Neural Networks,
[Phys. Rev. B 104, 035120 (2021)](https://doi.org/10.1103/PhysRevB.104.035120).

