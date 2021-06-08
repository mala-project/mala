# Setting up LAMMPS

## Installing LAMMPS

* checkout <https://github.com/athomps/lammps/tree/compute-grid>
* Make sure the compute-grid tree is checked out!
* Compile the LAMMPS shared library with the SNAP package installed
    - cd into `/path/to/lammps/src` folder of LAMMPS
    - `make yes-snap`
    - (optional: check with `make ps` that snap was correctly added)
    - `make mode=shlib mpi`

## Make the LAMMPS libray visible

### Method 1 (links, not recommended)

* Make the shared library visible e.g. `ln -s $LAMMPS/src/liblammps.so /path/to/repo/src/datahandling`
* Make the LAMMPS `lammps.py` visible e.g. `ln -s $LAMMPS/python/lammps.py /path/to/repo/src/datahandling`

### Method 2 (package, recommended)

* `python3 install.py -m lammps.py -l ../src/liblammps.so -v ../src/version.h`
