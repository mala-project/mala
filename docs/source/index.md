![image](./img/logos/mala_horizontal.png)

# Welcome to MALA!

## What is MALA?

MALA is a software package for building ML models that replace
density functional theory (DFT) calculations. DFT itself is **the** most
popular method for simulating materials on a quantum level and predicting 
materials properties. Compared to other quantum mechanical simulation 
techniques, DFT simulations can be run with relatively modest computational 
ressources. Thus, DFT simulations are the backbone of computational materials 
science. 

However, when tackling, e.g., matter at large temperatures or generally trying 
to simulate materials at extended length scales, the comparably low 
computational effort of DFT simulations becomes insurmountable. DFT scales 
cubically with system size and temperature - simulations above a few thousand 
atoms or at millions of Kelvin are virtually impossible. 

Therefore, a new direction of research tries to [combine the power of ML with DFT](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.6.040301).
MALA is at the forefront of these efforts, being the first library to allow 
users to build ML-DFT models that give **full access to the electronic structure**,
observables such as the total free energy of the system **and** scale well with
system size and temperature.

## Who is MALA?

MALA is jointly developed by the [Sandia National Laboratories](https://www.sandia.gov/)
and the [Center for Advanced Systems Understanding](https://www.casus.science/).

## How is MALA? 




## Where is MALA? 

MALA has been used in a range of publications:

Published
*********

- [Accelerating finite-temperature Kohn-Sham density functional theory with deep neural networks](https://www.doi.org/10.1103/PhysRevB.104.035120) (Phys. Rev. B)
  by J. A. Ellis, L. Fiedler, G. A. Popoola, N. A. Modine, J. A. Stephens, A. P. Thompson, A. Cangi, and S. Rajamanickam

  - Original publication detailling MALA workflow and application to aluminium data at room temperature and across the melting point.
- 
- [Predicting electronic structures at any length scale with machine learning](https://doi.org/10.1038/s41524-023-01070-z) (npj Comput. Mater.)
  by L. Fiedler, N. A. Modine, S. Schmerler, D. J. Vogel, G. A. Popoola, A. P. Thompson, S. Rajamanickam, A. Cangi

  - This work shows how MALA can be used to perform size extrapolation, i.e., train models on small numbers of atoms and then perform inference on
    (ultra-)large scales. It predicts the electronic structure of over 100'000 atoms.

- [Training-free hyperparameter optimization of neural networks for electronic structures in matter](https://doi.org/10.1088/2632-2153/ac9956) (Mach. Learn.: Sci. Technol.)
  by L. Fiedler, N. Hoffmann, P. Mohammed, G. A. Popoola, T. Yovell, V. Oles, J. A. Ellis, S. Rajamanickam, A. Cangi

  - This work details how different hyperparameter optimization techniques employed in MALA influence surrogate model accuracy and how training-free metrics can
    improve the perfomance of identifying models and calculation bispectrum descriptors.

- [Accelerating Equilibration in First-Principles Molecular Dynamics with Orbital-Free Density Functional Theory](https://doi.org/10.1103/PhysRevResearch.4.043033) (Phys. Rev. Research)
  by L. Fiedler, Z. A. Moldabekov, X. Shao, K. Jiang, T. Dornheim, M. Pavanello, A. Cangi

  - This work shows how orbital-free DFT calculations can be used as input for MALA investigations to increase accuracy of predictions; further introduces
    a metric for determination of DFT-MD trajectory equilibration implemented in MALA.


Pre-Prints
***********





## Contents

```{toctree}
---
maxdepth: 2
---

installation
usage
api/modules
CONTRIBUTE
```

# Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
