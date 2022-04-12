External modules
================

MALA can be coupled to external libraries for data pre- and postprocesing.
The DFT code QuantumESPRESSO is used for the calculation of the density
contributions to the total energy, while the LAMMPS code is used to calculate
SNAP descriptors on the real space grid of a simulation cell, either for
training or inference.

QuantumESPRESSO (total energy module)
*************************************

The total energy module has been moved to its own fork of QE.
You can find it at
https://gitlab.com/casus/q-e/-/tree/tem_original_development after
gaining access from the MALA development team. After checking out this
fork and branch, go to ``total_energy_module/`` and follow the installation
instructions described therein.


LAMMPS (descriptor calculation)
*******************************

* Checkout https://github.com/athomps/lammps/tree/compute-grid-new
* Make sure the compute-grid tree is checked out!
* Compile the LAMMPS shared library with the SNAP package installed

  - cd into ``/path/to/lammps/src`` folder of LAMMPS
  - ``make yes-ml-snap``
  - (optional: check with ``make ps`` that ``ml-snap`` was correctly added)
  - ``make mode=shlib mpi``

* Make the LAMMPS library visible via ``python3 install.py -p lammps -l ../src/liblammps.so -v ../src/version.h``
