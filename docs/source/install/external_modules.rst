External modules
================

MALA can be coupled to external libraries for data pre- and postprocesing.
The DFT code Quantum ESPRESSO is used for the calculation of the density
contributions to the total energy, while the LAMMPS code is used to calculate
bispectrum descriptors on the real space grid of a simulation cell, either for
training or inference.

Quantum ESPRESSO (total energy module)
***************************************

To run the total energy module, you need a full Quantum ESPRESSO (version
6.4.1.) and install the python bindings for it.

Build Quantum ESPRESSO
-----------------------

* Clone QE source code: ``git@gitlab.com:QEF/q-e.git``
* Chage into the ``q-e`` directory and check out the correct branch
  (the total energy module is based on version 6.4.1): ``git checkout qe-6.4.1``
* Make sure you have an (MPI-aware) F90 compiler such as ``mpif90`` (e.g.
  Debian-ish machine: ``apt install openmpi-bin``, on an HPC cluster something
  like ``module load openmpi gcc``). Make sure to use the same compiler
  for QE and the extension (``--f90exec=`` in ``build_total_energy_energy_module.sh``).
* We assume that QE's ``configure`` script will find your system libs, e.g. use
  ``-lblas``, ``-llapack`` and ``-lfftw3``. We use those by default in
  ``build_total_energy_energy_module.sh``. If you have, say, the MKL library,
  you may see ``configure`` use something like ``-lmkl_intel_lp64 -lmkl_sequential -lmkl_core``
  when building QE. In this case you have to modify
  ``build_total_energy_energy_module.sh`` to use the same libraries!

  * GNU compiler specific: we use ``-fallow-argument-mismatch``
* Change to the ``external_modules/total_energy_module`` directory of the
  MALA repository
* Run the script ``prepare_qe.sh /path/to/your/q-e`` with ``/path/to/your/qe``
  being the path to the ``q-e`` directory
* Change to the ``q-e`` directory

  * If you already have a build of QE, go into the ``q-e`` folder and run ``make veryclean``.
* Run ``./configure CFLAGS="-fPIC" CPPFLAGS="-fPIC" FFLAGS="-fPIC -fallow-argument-mismatch"``
* Run ``make all`` (use ``make -j<your number of cores> all`` for a faster
  compilation process).
* Change back to the  ``external_modules/total_energy_module`` directory of the
  MALA repository
* Run ``build_total_energy_energy_module.sh /path/to/your/q-e``.

  * If the build is successful, a file named something like
    ``total_energy.cpython-39m-x86_64-linux-gnu.so`` will be generated. This is
    the Python extension module.

Use and test the Python extension module
------------------------------------------

* Add the ``external_modules/total_energy_module`` directory to your python
  path, e.g. via ``export PYTHONPATH=/path/to/mala/external_modules/total_energy_module:$PYTHONPATH``
* Now you can use ``import total_energy`` to access the total energy module
* The MALA test suite will test the total energy module

LAMMPS (descriptor calculation)
*******************************

We provide a LAMMPS version compatible with the most recent MALA version
`here <https://github.com/mala-project/lammps/tree/mala>`_.

Building the LAMMPS interface for MALA consists of two steps:

1. Build a LAMMPS instance
2. Build the python bindings to it

The second part is mostly trivial. The first part is a bit more involved, since
there are few choices to be made here. For a full overview of how to build
LAMMPS, please refer to the `official instructions <https://docs.lammps.org/Build.html>`_.
The MALA team recommends to build LAMMPS with ``cmake``. To do so

* Checkout https://github.com/mala-project/lammps/tree/mala
* Make sure the ``mala`` tree is checked out locally via ``git branch``!
* Inside the LAMMPS folder create a build folder (named, e.g., ``build``)
* In the ``build`` folder, configure your ``cmake`` build:
  ``cmake ../cmake -D OPTION1 -D OPTION2 ...``; Options for a typical LAMMPS
  build for usage with MALA:

  * ``BUILD_SHARED_LIBS=yes``: Necessary to link the LAMMPS library to python
  * ``PKG_ML-SNAP=yes``: Enables the calculation of SNAP descriptors with LAMMPS
  * ``BUILD_MPI=yes``: Enables MPI aware calculations; set this option if
    plan to use MPI and have an MPI aware compiler loaded
  * ``PKG_KOKKOS=yes``: Enables the Kokkos package which is needed to calculate
    bispectrum descriptors on a GPU. Without this option, bispectrum descriptor
    calculation will solely be performed on CPU. If you want to use a GPU via
    Kokkos, further options you will have to set further options:

      * ``Kokkos_ENABLE_CUDA=yes``: Tells Kokkos to use CUDA
      * ``Kokkos_ARCH_HOSTARCH=???``: Your CPU architecture (see `Kokkos instructions <https://docs.lammps.org/Build_extras.html#kokkos-package>`_)
      * ``Kokkos_ARCH_GPUARCH=???``: Your GPU architecture (see see `Kokkos instructions <https://docs.lammps.org/Build_extras.html#kokkos-package>`_)
      * ``CMAKE_CXX_COMPILER=???``: Path to the ``nvcc_wrapper`` executable
        shipped with the LAMMPS code, should be at ``/your/path/to/lammps/lib/kokkos/bin/nvcc_wrapper``

* Build the library and executable with ``cmake --build .``
  (Add ``--parallel=8`` for a faster build)

This should create a shared library called ``liblammps.so`` in your build
folder. To then build the python library

* Change into the ``python`` folder of the LAMMPS code
* ``python3 install.py -p lammps -l ../<build_folder>/liblammps.so``, where
  ``<build_folder>`` is whatever folder you performed the ``cmake`` build in
* Note: The python installation process may give an ``shutil.SameFileError``
  after successful installation; this is LAMMPS related and can be ignored
  here.
