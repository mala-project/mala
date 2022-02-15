# Overview

## List of features

* General features

  * Central parameters class that holds all necessary parameters and allows for saving and loading for later use

  * Interfaces to MPI and horovod for parallel training and inference

* Preprocesing of simulation data

  * Calculation of atomic descriptors from simulation data

  * Parsing of LDOS data files

  * Scaling and Conversion of data


* Training of Surrogate models

  * Creation, training and evaluation of feed-forward neural networks

  * Training progress can be checkpointed, neural networks can be saved for later use

  * Distributed training with horovod (experimental)

  * (Distributed) hyperparameter optimization with optuna

  * Hyperparameter optimization with orthogonal array tuning and neural architecure search without training

* Postprocessing of surogate model output

  * LDOS can be used to calculate DOS and/or density

  * Different quantities of interest can be calculated

    * Number of electrons (from LDOS, DOS or density)

    * Band energy (from LDOS or DOS)

    * Total energy (requires QE; from LDOS or DOS + density)

    * Radial distribution function

    * Three particle correlation function

    * Static structure factor


## Getting started

* [Installation instructions](../installation.rst)

* [Examples](https://github.com/mala-project/mala/tree/develop/examples)

* [Usage overview](../usage.rst)

* Further reading:

  * [Accelerating finite-temperature Kohn-Sham density functional theory with deep neural networks](https://doi.org/10.1103/PhysRevB.104.035120)
