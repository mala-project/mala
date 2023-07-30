Installing Quantum ESPRESSO (total energy module)
=================================================

Prerequisites
*************

To run the total energy module, you need a full Quantum ESPRESSO installation,
for which to install the Python bindings. This module has been tested with
version ``7.2.``, the most recent version at the time of this release of MALA.
Newer versions may work (untested), but installation instructions may vary.

Make sure you have an (MPI-aware) F90 compiler such as ``mpif90`` (e.g.
Debian-ish machine: ``apt install openmpi-bin``, on an HPC cluster something
like ``module load openmpi gcc``). Make sure to use the same compiler
for QE and the extension. This should be the default case, but if problems
arise you can manually select the compiler via
``--f90exec=`` in ``build_total_energy_energy_module.sh``

We assume that QE's ``configure`` script will find your system libs, e.g. use
``-lblas``, ``-llapack`` and ``-lfftw3``. We use those by default in
``build_total_energy_energy_module.sh``. If you have, say, the MKL library,
you may see ``configure`` use something like ``-lmkl_intel_lp64 -lmkl_sequential -lmkl_core``
when building QE. In this case you have to modify
``build_total_energy_energy_module.sh`` to use the same libraries!

Build Quantum ESPRESSO
**********************

* Download QE 7.2: ``https://gitlab.com/QEF/q-e/-/releases/qe-7.2``
* Change to the main QE directory (default: ``q-e-qe-7.2``, you can rename this
  directory as you wish)
* Run ``./configure CFLAGS="-fPIC" CPPFLAGS="-fPIC" FFLAGS="-fPIC"``
* Run ``make all`` (use ``make -j<your number of cores> all`` for a faster
  compilation process).
* Change to the  ``external_modules/total_energy_module`` directory of the
  MALA repository

Installing the Python extension
********************************

* Run ``build_total_energy_energy_module.sh /path/to/your/q-e``.

  * If the build is successful, a file named something like
    ``total_energy.cpython-39m-x86_64-linux-gnu.so`` will be generated. This is
    the Python extension module.
* Add the ``external_modules/total_energy_module`` directory to your Python
  path, e.g. via ``export PYTHONPATH=/path/to/mala/external_modules/total_energy_module:$PYTHONPATH``
* Now you can use ``import total_energy`` to access the total energy module
* The MALA test suite will test the total energy module
