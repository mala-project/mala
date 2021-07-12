# ML-DFT

This is the original code used in [1].

# ML-DFT


ML-DFT is a machine learning framework to that accelerated firt principles calculations (density functional theory) using novel algorithms. The framework is made available open sourcei for community engagement. See License and Copyright files for further information.

ML-DFT is designed open-source python package with dependencies on pyTorch and Horovod. It enables users to perform the entire modeling toolchain using only a few lines of code. ML-DFT is jointly developed by the Sandia National Laboratories (SNL), and the Center for Advanced Systems Understanding (CASUS). 
This repository is structured as follows:
```
├── descriptors : contains code for SNAP descriptors
├── networks : contains the deep neural network models. See [Network README](networks/README.md) for more information
├── notenooks : contains python notebooks and post processing scripts
├── total_energy_module : Changes and steps needed to use Quantum Espresso to generate total energy.
```


## Institutions
### Founding Institutions

- Sandia National Laboratories (SNL), USA.
- Center for Advanced Systems Understandingi (CASUS), Germany.

## Developers
### Scientific Supervision
- Attila Cangi (CASUS)
- Siva Rajamanickam (SNL)

### Core Developers
- Attila Cangi (CASUS)
- Austin Ellis (ORNL*)
- Normand Modine (SNL)
- Gabriel Popoola (SNL)
- Siva Rajamanickam (SNL)
- Adam Stephens (SNL)
- Aidan Thompson (SNL)

* - Work done as part of postdoctoral research at Sandia National Laboratories


## Citing ML-DFT

If you publish work which uses or mentions MALA, please cite the following paper:

J. A. Ellis, L. Fiedler, G. A. Popoola, N. A. Modine, J. A. Stephens, A. P. Thompson,
A. Cangi, S. Rajamanickam (2020). Accelerating Finite-temperature
Kohn-Sham Density Functional Theory with Deep Neural Networks.
[arXiv:2010.04905](https://arxiv.org/abs/2010.04905).
