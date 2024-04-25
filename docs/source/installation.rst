Installation
============

As a software package, MALA consists of three parts:

1. The actual Python package ``mala``, which this documentation accompanies
2. The `Quantum ESPRESSO <https://www.quantum-espresso.org/>`_ (QE) code, which
   is used by MALA to post-process the LDOS into total free energies (via the
   so called "total energy module")
3. The `LAMMPS <https://www.lammps.org/>`_ code, which is used by MALA to
   encode atomic structures on the real-space grid (optional, but highly
   recommended!)

All three parts require separate installations. The most important one is
the first one, i.e., the Python library, and you can access a lot of MALA
functionalities by just installing the MALA Python library, especially when
working with precalculated input and output data (e.g. for model training).

For access to all feature, you will have to furthermore install the QE code.
The calculations performed by LAMMPS are also implemented in the python part
of MALA. For small test calculations and development tasks, you therefore do
not need LAMMPS. For realistic simulations the python implementation is not
efficient enough, and you have to use LAMMPS.

The installation has been tested on Linux (Ubuntu/CentOS), Windows and macOS.
The individual installation steps are given in:

.. toctree::
   :maxdepth: 1

   install/installing_mala
   install/installing_qe
   install/installing_lammps
