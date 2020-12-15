# Installation of ML-DFT@CASUS

This document details the manual installation of the ML-DFT package. 

## Prerequisites

### Python packages 

In order to run ML-DFT you have to have the following packages installed :

* torch (PyTorch)
* numpy
* scipy
* optuna
* ase

You can install python packages with 

```
pip install packagename 
```

For the installation of PyTorch you might also refer to https://pytorch.org/.

###  LAMMPS (optional)

This workflow uses the Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS, https://lammps.sandia.gov/) for input data preprocessing. It is not necessary to install LAMMPS if you are you using this workflow with preprocessed data. If you need/want to install LAMMPS, please refer to INSTALL_SNAP.md.

## Installing the ML-DFT package

1. Make sure that all prerequisites are fulfilled (see above)
2. Copy the setup.py file from this directory into the git root dirdctory

    ```
    cd ~/path/to/this/git/root/directory
    cp examples/setup.py .
    ```

3. Install the package using 

    ```
    cd ~/path/to/this/git/root/directory
    pip install -e .
    ```
    (note: the ```-e``` is absolutely crucial, so that changes in the code will be reflected system wide)
4. Go to the ```examples``` folder and run ```ex0_verify_installation.py``` to make sure the setup was successful. 

5. Enjoy!




