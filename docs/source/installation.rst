Installation
============

As a software package, MALA consists of three parts:

1. The actual Python package ``mala``, which this documentation accompanies
2. The `LAMMPS <https://www.lammps.org/>`_ code, which is used by MALA to
   encode atomic structures on the real-space grid
3. The `Quantum ESPRESSO <https://www.quantum-espresso.org/>`_ (QE) code, which
   is used by MALA to post-process the LDOS into total free energies (via the
   so called "total energy module")

All three parts require separate installations. The most important one is
the first one, i.e., the Python library, and you can access a lot of MALA
functionalities by just installing the MALA Python library, especially when
working with precalculated input and output data (e.g. for model training).

For access to all feature, you will have to furthermore install the LAMMPS
and QE codes and associated Python bindings. The installation has been tested
on Linux (Ubuntu/CentOS), Windows and macOS. The individual installation steps
are given in:

.. toctree::
   :maxdepth: 1

   install/installing_mala
   install/installing_lammps
   install/installing_qe
