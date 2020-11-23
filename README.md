# ml-dft-casus

WIP: Building of a ML-DFT framework based on the one used in cooperation with Sandia, but with more usability and features.

## ToDo - Rewrite

* Create sceleton code
    * use cli_test.py to test the progress of the framework
    * later also write a .yaml file for CI
* Implement the interface to LAMMPS to calculate SNAP descriptors
* Implement NN driver (and make it modular)
* Implement a nice inference method

## ToDo - future+maybeNotPossible

* Find a way to calculate LDOS from this python module
* find a way to do some QE stuff from this module
* switch to different, python based data generating code (e.g. GPAW) that is easily accessible
