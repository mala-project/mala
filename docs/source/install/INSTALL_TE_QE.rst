Build Python Module to run QE “v of rho” subroutines
====================================================

All the files described here can be found in the subfolder
``install/total_energy_module/``.

Using GNU and the mlmm_env environment on Blake
-----------------------------------------------

Initial Setup
~~~~~~~~~~~~~

1. Clone QE source: ``git clone https://github.com/QEF/q-e.git`` On
   blake, it may be necessary to ‘module load git’ first.
2. Change into the ``q-e`` directory and check out out the qe-6.4.1 tag:
   ``git checkout qe-6.4.1``
3. Make sure you have the necessary modules loaded (``gcc``, ``python``,
   ``openmpi``).
4. Patch the QE source so that the FoX project is built with ``-fPIC``:

   1. change to the ``q-e/install/m4`` folder
   2. ``patch < x_ac_qe_f90.m4.patch``

5. Copy ``total_energy.f90`` to ``q-e/PW/src``. This file contains the
   Fortran 90 code that ``f2py`` will create a Python binding to.

Build Quantum Espresso
~~~~~~~~~~~~~~~~~~~~~~

1.  If you already have a build of QE, go into the ``q-e`` folder and
    run ``make veryclean``.
2.  Make sure you have the necessary modules loaded (``gcc``,
    ``openmpi``).
3.  Set up the environment using
    ``source total_energy_module_env_gnu.sh.``
4.  In order for this module to work, your QE installation has to be
    linked against custom compiled LAPACK and BLAS libraries. If you
    have LAPACK/BLAS already installed somewhere on your system, QE will
    use these installations; they will have to be recompiled correctly
    (see below)

    1. A simple workaround is to force QE to build its own LAPACK and
       BLAS.
    2. In order to do that open ``q-e/install/make.inc.in`` uncomment
       the line ``BLAS_LIBS_SWITCH = internal`` and the line above;
       comment the line ``BLAS_LIBS_SWITCH = external`` and the line
       above.
    3. Do the same for the LAPACK lines.
    4. Both for LAPACK and BLAs, make sure to add a ``lib`` to the
       ``*.a`` files (e.g. ``lapack.a`` to ``liblapack.a``

5.  Add ``-fallow-argument-mismatch`` at the end of line 105
    (``F90FLAGS``) in ``q-e/install/make.inc.in``
6.  Run ``./configure`` in the q-e directory.
7.  Afterwards, configure the make input for LAPACK:

    1. Add ``-fallow-argument-mismatch`` to line 20 (``OPTS``) in
       ``q-e/install/make_lapack.inc``
    2. Add ``-fPIC`` to line 22 (``NOOPT``) in
       ``q-e/install/make_lapack.inc``

8.  Run ``make all`` (use ``make -j8 all`` for a faster compilation
    process)

    1. I (L. Fiedler) had to clean and recompile in the middle of the
       compilation process; that might be due to a problem with the
       parallel build I was doing. Maybe serial build is not the worst
       idea here.

9.  In ``q-e/LAPACK``, run ``make clean`` and ``make``
10. In ``q-e/LAPACK/CBLAS``, run ``make clean`` and ``make``.
11. The FoX library included in QE is delivered precompiled; this omits
    the compilation with the ``-fPIC`` option necessary for usage of the
    total energy module.

    1. Download the FoX library from https://github.com/andreww/fox
    2. Compile the library, make sure that the ``-fPIC`` compiler option
       is used (this should already be the case after step 3)
    3. Replace the ``q-e/FoX/lib/`` folder with the ``fox/objs/lib``
       folder
    4. Replace the ``q-e/FoX/finclude/`` folder with the
       ``fox/objs/finclude`` folder

Build the Python Module
~~~~~~~~~~~~~~~~~~~~~~~

1. Change to ``q-e/PW/src`` and run the
   ``build_total_energy_energy_module_gnu.sh`` script in the
   ``total/energy`` folder. If the build is successful, a file named
   ``total_energy.cpython-37m-x86_64-linux-gnu.so`` will be generated.
   It contains the Python module.

Use the Python Module
~~~~~~~~~~~~~~~~~~~~~

1. Add the folder that contains
   ``total_energy.cpython-37m-x86_64-linux-gnu.so`` to your
   ``PYTHONPATH``.
2. Now you can use: ``import total_energy``

Using Intel/MKL
---------------

.. warning::
   Not tested/supported yet! Please contact the authors to receive the original files from the Sandia
   National Laboratories group.
