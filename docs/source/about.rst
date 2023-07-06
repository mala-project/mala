About
=====

What is MALA? MALA is a software package for building ML models that replace
density functional theory (DFT) calculations. DFT itself is **the** most
popular simulation method for simulating materials on a quantum level.
Through DFT simulations, one can gain insights in a range of materials
properties without any requirements for empirical data at all.

DFT simulations are routinely employed by researchers around the world, since
they allow for

Publications
------------

Here's an overview of publications that have used MALA in the past.

Published
*********

- `Accelerating finite-temperature Kohn-Sham density functional theory with deep neural networks <https://www.doi.org/10.1103/PhysRevB.104.035120>`_ (Phys. Rev. B)
  by J. A. Ellis, L. Fiedler, G. A. Popoola, N. A. Modine, J. A. Stephens, A. P. Thompson, A. Cangi, and S. Rajamanickam

  - Original publication detailling MALA workflow and application to aluminium data at room temperature and across the melting point.

- `Training-free hyperparameter optimization of neural networks for electronic structures in matter <https://doi.org/10.1088/2632-2153/ac9956>`_ (Mach. Learn.: Sci. Technol.)
  by L. Fiedler, N. Hoffmann, P. Mohammed, G. A. Popoola, T. Yovell, V. Oles, J. A. Ellis, S. Rajamanickam, A. Cangi

  - This work details how different hyperparameter optimization techniques employed in MALA influence surrogate model accuracy and how training-free metrics can
    improve the perfomance of identifying models and calculation bispectrum descriptors.

- `Accelerating Equilibration in First-Principles Molecular Dynamics with Orbital-Free Density Functional Theory <https://doi.org/10.1103/PhysRevResearch.4.043033>`_ (Phys. Rev. Research)
  by L. Fiedler, Z. A. Moldabekov, X. Shao, K. Jiang, T. Dornheim, M. Pavanello, A. Cangi

  - This work shows how orbital-free DFT calculations can be used as input for MALA investigations to increase accuracy of predictions; further introduces
    a metric for determination of DFT-MD trajectory equilibration implemented in MALA.


Pre-Prints
***********

- `Predicting electronic structures at any length scale with machine learning <https://doi.org/10.48550/arXiv.2210.11343>`_
  by L. Fiedler, N. A. Modine, S. Schmerler, D. J. Vogel, G. A. Popoola, A. P. Thompson, S. Rajamanickam, A. Cangi

  - This work shows how MALA can be used to perform size extrapolation, i.e., train models on small numbers of atoms and then perform inference on
    (ultra-)large scales. It predicts the electronic structure of over 100'000 atoms.
