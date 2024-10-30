.. _lammpsinstallation:

Installing LAMMPS
==================

Prerequisites
**************

Make sure you have ``cmake`` and suitable compilers (e.g. ``gcc``) available.
LAMMPS can be build with GPU support (recommended for production settings!),
in which case `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ is required
to be installed on your machine.

Build LAMMPS
************

We provide a LAMMPS version compatible with the most recent MALA version
`here <https://github.com/mala-project/lammps/tree/mala>`_, which should always
be used with MALA. For a full overview of how to build LAMMPS, please refer to
the `official instructions <https://docs.lammps.org/Build.html>`_.
The MALA team recommends to build LAMMPS with ``cmake``. To do so

* Checkout https://github.com/mala-project/lammps/tree/mala
* Make sure the ``mala`` tree is checked out locally via ``git branch``!
* Inside the LAMMPS folder create a build folder (named, e.g., ``build``)
* In the ``build`` folder, configure your ``cmake`` build:
  ``cmake ../cmake -D OPTION1 -D OPTION2 ...``; Options for a typical LAMMPS
  build for usage with MALA:

  * ``BUILD_SHARED_LIBS=yes``: Necessary to link the LAMMPS library to Python
  * ``PKG_ML-SNAP=yes``: Enables the calculation of SNAP descriptors with LAMMPS
  * ``BUILD_MPI=yes``: Enables MPI aware calculations; set this option if
    plan to use MPI and have an MPI aware compiler loaded
  * ``PKG_KOKKOS=yes``: Enables the Kokkos package which is needed to calculate
    bispectrum descriptors on a GPU. Without this option, bispectrum descriptor
    calculation will solely be performed on CPU. If you want to use a GPU via
    Kokkos (recommended for production settings!), you will have to set further options:

      * ``Kokkos_ENABLE_CUDA=yes``: Tells Kokkos to use CUDA
      * ``Kokkos_ARCH_HOSTARCH=???``: Your CPU architecture (see `Kokkos instructions <https://docs.lammps.org/Build_extras.html#kokkos-package>`_)
      * ``Kokkos_ARCH_GPUARCH=???``: Your GPU architecture (see see `Kokkos instructions <https://docs.lammps.org/Build_extras.html#kokkos-package>`_)
      * ``CMAKE_CXX_COMPILER=???``: Path to the ``nvcc_wrapper`` executable
        shipped with the LAMMPS code, should be at ``/your/path/to/lammps/lib/kokkos/bin/nvcc_wrapper``

    For example, this configures the LAMMPS cmake build with Kokkos support
    for an Intel Haswell CPU and an Nvidia Volta GPU, with MPI support:

      .. code-block:: bash

            cmake ../cmake -D PKG_KOKKOS=yes -D BUILD_MPI=yes -D PKG_ML-SNAP=yes -D Kokkos_ENABLE_CUDA=yes -D Kokkos_ARCH_HSW=yes -D Kokkos_ARCH_VOLTA70=yes -D CMAKE_CXX_COMPILER=/path/to/lammps/lib/kokkos/bin/nvcc_wrapper -D BUILD_SHARED_LIBS=yes

.. note::
      When using a GPU by setting ``parameters.use_gpu = True``, you *need* to
      have a GPU version of ``LAMMPS`` installed. See :ref:`production_gpu` for
      details.

* Build the library and executable with ``cmake --build .``
  (Add ``--parallel=8`` for a faster build)



Installing the Python extension
********************************


* After successfully having built LAMMPS, Change into the ``python`` folder of the LAMMPS code
* ``python3 install.py -p lammps -l ../<build_folder>/liblammps.so -v ../src/version.h``, where
  ``<build_folder>`` is whatever folder you performed the ``cmake`` build in
* Note: The Python installation process may give an ``shutil.SameFileError``
  after successful installation; this is LAMMPS related and can be ignored
  here.
