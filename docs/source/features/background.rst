Background
===========

Workflow
*********

The goal of FESL is to build surrogate models for electronic structure theory.
These surrogate models are based on neural networks. After training such
a model, it allows the fast evaluation of the total energy and atomic forces.
FESL is build around Density Functional Theory, but can in
principle be used with all electronic structure methods that calculate the
total energy and atomic forces given atomic positions as input.
Building these surrogate models requires preprocessing
:doc:`preprocessing <preprocessing>` of the data, training of a
:doc:`neural network <neuralnetworks>` and
:doc:`postprocessing <postprocessing>` of the results.
FESL is designed for the investigation of systems at non-zero temperatures and
operates in a "per-grid-point" manner, meaning that every grid point of a
simulation cell is passed through the network individually.

Density Functional Theory
*************************

Density Functional Theiry is one of the most popular electronic structure
calculation methods due to its combination of reasonable accuracy and
computational cost.
In DFT, the central quantity is the electronic density of a given system.
Within the Kohn-Sham framework, this density is given by

.. math::

    n(\boldsymbol{r}) = \sum_j f^\beta(\epsilon_j)\,
    |\phi_j(\boldsymbol{r})|^2 \; .

Here, :math:`\phi_j(\boldsymbol{r})` denote the Kohn-Sham wave functions,
which are given by the Kohn-Sham equations

.. math::

   \left[-\frac{1}{2}\nabla^2 + v_\mathrm{{\scriptscriptstyle S}}(\mathbf{r};
   \underline{\boldsymbol{R}})\right] \phi_j(\boldsymbol{r};
   \underline{\boldsymbol{R}}) = \epsilon_j \phi_j(\boldsymbol{r};
   \underline{\boldsymbol{R}}) \; ,

which give a system of non-interacting particles restricted to reproduce
the density of the interacting system. The total energy is evaluated using

.. math::

    E_\mathrm{total}(\underline{\boldsymbol{r}}) =
    T_\mathrm{{\scriptscriptstyle S}}
    [n](\underline{\boldsymbol{r}}) -
    S_\mathrm{{\scriptscriptstyle S}}
    [n](\underline{\boldsymbol{r}})/\beta
    + E_\mathrm{{\scriptscriptstyle H}}
    [n](\underline{\boldsymbol{r}}) +
    E_\mathrm{{\scriptscriptstyle XC}}[n](\underline{\boldsymbol{r}})
    + E^{ei}[n](\underline{\boldsymbol{r}})+ E^{ii} + \mu N_e \; .

Forces and other quantities of interest can be derived from the total energy.
