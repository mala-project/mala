![image](./img/logos/mala_horizontal.png)

# Welcome to MALA!

## What is MALA?

MALA is a software package for building ML models that replace
density functional theory (DFT) calculations. DFT is one of the most widely 
used methods for simulating materials at a quantum level and predicting their 
properties, employed by researchers worldwide.

Although DFT is more efficient compared to other quantum mechanical simulation 
techniques, it still exhibits cubic scaling with the number of atoms and 
temperature. This limitation poses significant computational challenges for 
conducting large-scale simulations.

Recent research activities focus on harnessing the potential of 
[ML and DFT](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.6.040301).
MALA leads these efforts by facilitating the construction of ML-DFT models, 
offering comprehensive access to the electronic structure and observables, 
including the total free energy of the system. Moreover, MALA ensures 
scalability with system size and various parameters, such as temperature.

## How does MALA work? 

MALA operates by predicting the so called local density of states (LDOS). 
The LDOS is an efficient and versatile representation of the electronic 
structure of a material. Based on the LDOS, a range of useful observables can
be calculated. 

MALA calculates a grid-based representation for any given atomic structure 
(so called descriptors). These descriptors are subsequently fed into a neural 
network, an efficient ML algorithm, to predict the LDOS.
The prediction is performed individually at each point of the real space grid, 
facilitating straightforward scalability and parallelization.
 

Subsequently, MALA offers a variety of post-processing features to compute 
desired observables. Similar to a DFT calculation, but with enhanced speed. 
For further details on the workflow, refer to the 
[first MALA publication](https://www.doi.org/10.1103/PhysRevB.104.035120).


## Where to start?

You can effortlessly set up MALA on your system by following the [installation instructions.](./installation.rst)
To get acquainted with MALA, explore the [basic usage guide](./basic_usage.rst),
along with the example files in the repository. 
Additionally, MALA offers various options for [advanced use](./advanced_usage.rst), 
allowing you to enhance training and inference speed, scale to large systems, 
and visualize results. 

If you use MALA in your research, please make sure to [cite accordingly](./citing.rst).


## Who is behind MALA?

MALA is jointly developed by the [Sandia National Laboratories](https://www.sandia.gov/)
(Scientific supervisor: Sivasankaran Rajamanickam, code maintenance: 
Jon Vogel)
and the [Center for Advanced Systems Understanding](https://www.casus.science/), 
an institute of the [Helmholtz-Zentrum Dresden-Rossendorf](https://www.hzdr.de/) 
(Scientific supervisor: Attila Cangi, code maintenance: Lenz Fiedler).

## MALA publications 

MALA has been employed in various publications, showcasing its versatility and effectiveness.

- [Accelerating finite-temperature Kohn-Sham density functional theory with deep neural networks](https://www.doi.org/10.1103/PhysRevB.104.035120) (Phys. Rev. B)
  by J. A. Ellis, L. Fiedler, G. A. Popoola, N. A. Modine, J. A. Stephens, A. P. Thompson, A. Cangi, and S. Rajamanickam

  - First MALA publication, introduces the workflow and shows proof-of-concept 
    resultsfor Aluminium at room temperature and across the melting point

- [Training-free hyperparameter optimization of neural networks for electronic structures in matter](https://doi.org/10.1088/2632-2153/ac9956) (Mach. Learn.: Sci. Technol.)
  by L. Fiedler, N. Hoffmann, P. Mohammed, G. A. Popoola, T. Yovell, V. Oles, J. A. Ellis, S. Rajamanickam, A. Cangi

  - This publication discusses hyperparameter optimization and shows how 
    the training-free surrogate metrics implemented in MALA can be used for 
    rapid identification of suitable hyperparameters.

- [Accelerating Equilibration in First-Principles Molecular Dynamics with Orbital-Free Density Functional Theory](https://doi.org/10.1103/PhysRevResearch.4.043033) (Phys. Rev. Research)
  by L. Fiedler, Z. A. Moldabekov, X. Shao, K. Jiang, T. Dornheim, M. Pavanello, A. Cangi

  - This publication shows how costly DFT-MD simulations can be accelerated 
    by initializing a simulation with OF-DFT. These results extend to ML-DFT 
    models, with MALA serving as an example.

- [Predicting electronic structures at any length scale with machine learning](https://doi.org/10.1038/s41524-023-01070-z) (npj Comput. Mater.)
  by L. Fiedler, N. A. Modine, S. Schmerler, D. J. Vogel, G. A. Popoola, A. P. Thompson, S. Rajamanickam, A. Cangi
  
  - This article asserts MALA's capability for size transfer; based on training
    data calculated for hundreds of atoms, MALA can predict the electronic
    structure of up to 100'000 atoms.

- [Machine learning the electronic structure of matter across temperatures](https://doi.org/10.1103/PhysRevB.108.125146) (Phys. Rev. B)
  by L. Fiedler, N. A. Modine, K. D. Miller, A. Cangi
 
  - This publication shows how MALA models can be employed across temperature 
    ranges. It is demonstrated how such models account for both ionic and
    electronic temperature effects of materials.




## Contents

```{toctree}
---
maxdepth: 1
---

installation
basic_usage
advanced_usage
citing
CONTRIBUTE
api/modules
```
