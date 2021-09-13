Overview
===========

List of features
****************

* General features

  * Central parameters class that holds all necessary parameters and allows for saving and loading for later use

* Preprocesing of simulation data

  * Calculation of atomic descriptors from simulation data

  * Parsing of LDOS data files

  * Scaling and Conversion of data


* Training of Surrogate models

  * Creation, training and evaluation of feed-forward neural networks

  * Training progress can be checkpointed, neural networks can be saved for later use

  * Distributed training with horovod (experimental)

  * (Distributed) hyperparameter optimization with optuna

  * Hyperparameter optimization with orthogonal array tuning and neural architecure search without training (both experimental)

* Postprocessing of surogate model output

  * LDOS can be used to calculate DOS and/or density

  * Different quantities of interest can be calculated

    * Number of electrons (from LDOS, DOS or density)

    * Band energy (from LDOS or DOS)

    * Total energy (requires QE; from LDOS or DOS + density)


Workflow
*********

The goal of MALA is to build surrogate models for electronic structure theory.
These surrogate models are based on neural networks. After training such
a model, it allows the fast evaluation of the total energy and atomic forces.
MALA is build around Density Functional Theory, but can in
principle be used with all electronic structure methods that calculate the
total energy and atomic forces given atomic positions as input.
Building these surrogate models requires preprocessing
:doc:`preprocessing <preprocessing>` of the data, training of a
:doc:`neural network <neuralnetworks>` and
:doc:`postprocessing <postprocessing>` of the results.
MALA is designed for the investigation of systems at non-zero temperatures and
operates in a "per-grid-point" manner, meaning that every grid point of a
simulation cell is passed through the network individually.
