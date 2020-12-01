# ml-dft-casus

WIP: Building of a ML-DFT framework based on the one used in cooperation with Sandia, but with more usability and features.
I have never programmed a ML framework before, so feel free to criticize my coding approach.

## Installation

__TODO__ Installation/setup script coming soon.

For now, see the install directory and follow the steps outlined there.
Make sure you have the following python modules installed:
* pytorch
* numpy
* scipy

Make sure you have lammps installed. For now that means:
    * https://github.com/athomps/lammps/tree/compute-grid
    * cd
    * Make sure that you compile it with mpi! For now, only mpi does work:
    * make mode=shlib mpi
    * Make the shared library visible e.g. ln -s $LAMMPS/src/liblammps.so src/datahandling
    * Make the LAMMPS lammps.py visible e.g. ln -s $LAMMPS/python/lammps.py src/datahandling

__TODO__ Make the lammps part of the code more accesible.
