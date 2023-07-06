![image](./img/logos/mala_horizontal.png)

# Welcome to MALA!

## What is MALA?

MALA is a software package for building ML models that replace
density functional theory (DFT) calculations. DFT is **the** most
popular method for simulating materials on a quantum level and predicting 
materials properties, and employed by researcher throughout the world. 

While DFT is very efficient compared to other quantum mechanical simulation
techniques, it still scales cubically with number of atoms and temperature 
simulated, making large scale simulation at arbitrary temperatures very 
computationally challenging. 

Therefore, a new direction of research tries to [combine the power of ML with DFT](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.6.040301).
MALA is at the forefront of these efforts, being the first library to allow 
users to build ML-DFT models that give **full access to the electronic structure**,
observables such as the total free energy of the system **and** scale well with
system size and temperature.

## How is MALA? 

MALA operates by predicting the so called local density of states (LDOS). 
The LDOS is an efficient and versatile representation of the electronic 
structure of a material. Based on the LDOS, a range of useful observables can
be calculated. 

MALA calculates a grid-based representation of any atomic structure presented 
to it (so called descriptors). These descriptors are then passed through a 
neural network, a very efficient ML algorithm, to predict the LDOS. This is 
done individually at each point of the real space grid, allowing for easy 
scalability and parallelization. 

Afterwards, a range of post-processing features are available within MALA to
calculate obervables from interest. Just like a DFT calculation - just faster.
For more detail on the workflow, check out the [first MALA publication](https://www.doi.org/10.1103/PhysRevB.104.035120)


## Where to start?

MALA can easily be set up on your system by following the [provided installation instructions.](./installation.rst)
To familiarize yourself with MALA, check out the [basic usage guide](./basic_usage.rst),
supplemented by the example files in the repository. 
MALA also provides ample options for [advanced use](./advanced_usage.rst), to, e.g., improve training
and inference speed, scale to large systems or visualize results. 


## Who is MALA?

MALA is jointly developed by the [Sandia National Laboratories](https://www.sandia.gov/)
and the [Center for Advanced Systems Understanding](https://www.casus.science/).

## Where is MALA? 

MALA has been used in a range of publications:

- [Accelerating finite-temperature Kohn-Sham density functional theory with deep neural networks](https://www.doi.org/10.1103/PhysRevB.104.035120) (Phys. Rev. B)
  by J. A. Ellis, L. Fiedler, G. A. Popoola, N. A. Modine, J. A. Stephens, A. P. Thompson, A. Cangi, and S. Rajamanickam

- [Predicting electronic structures at any length scale with machine learning](https://doi.org/10.1038/s41524-023-01070-z) (npj Comput. Mater.)
  by L. Fiedler, N. A. Modine, S. Schmerler, D. J. Vogel, G. A. Popoola, A. P. Thompson, S. Rajamanickam, A. Cangi

- [Training-free hyperparameter optimization of neural networks for electronic structures in matter](https://doi.org/10.1088/2632-2153/ac9956) (Mach. Learn.: Sci. Technol.)
  by L. Fiedler, N. Hoffmann, P. Mohammed, G. A. Popoola, T. Yovell, V. Oles, J. A. Ellis, S. Rajamanickam, A. Cangi

- [Accelerating Equilibration in First-Principles Molecular Dynamics with Orbital-Free Density Functional Theory](https://doi.org/10.1103/PhysRevResearch.4.043033) (Phys. Rev. Research)
  by L. Fiedler, Z. A. Moldabekov, X. Shao, K. Jiang, T. Dornheim, M. Pavanello, A. Cangi

- [Machine learning the electronic structure of matter across temperatures](https://doi.org/10.48550/arXiv.2306.06032) (arXiv preprint)
  by L. Fiedler, N. A. Modine, K. D. Miller, A. Cangi





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
