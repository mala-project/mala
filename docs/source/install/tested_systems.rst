Successfully tested on
=========================
Personal machines
*******************
Pop!_OS
---------------
* OS version: 20.10
* pip

  * python version: 3.8.6
  * Installation successful: Yes
* conda:

  * python version: 3.8.6
  * Installation successful: Yes
* LAMMPS: Yes, installed using :doc:`the instructions on external modules <external_modules>`
* Quantum Espresso: Yes, installed using :doc:`the instructions on external modules <external_modules>`

Ubuntu
---------------
* OS version: 20.10
* pip

  * python version: 3.8.6
  * Installation successful: Yes
* conda:

  * Installation successful: Not tested
* LAMMPS: Not tested
* Quantum Espresso: Not tested

macOS
---------------
* macOS 11.0.1 
* OS version: 20.10
* pip

  * Installation successful: Not tested
* conda:

  * python version: 3.8.5
  * Installation successful: Not tested


Windows
----------
* Windows 10
* OS Version: 10.0.19044 Build 19044
* pip:
  * pip version: 3.8.10
  * Installation successful: Yes
* conda:

  * Installation successful: Not tested
* LAMMPS: Not tested
* Quantum Espresso: Not tested


HPC clusters
***************
Hemera5 (CentOS)
-----------------

Hemera5 is the local cluster of the Helmholtz-Zentrum Dresden-Rossendorf
(one of the founding institutions of the MALA project).

.. warning:: Currently Quantum Espresso and LAMMPS have python bindings in different python version. They cannot be used
   within the same virtual environment. This issue will be adressed shortly.

* OS version: 7
* pip:

  * Installation successful: No, not available on hemera
* conda:

  * python version: 3.6.5 (to use Quantum Espresso), 3.8.0 (to use LAMMPS)
  * Installation successful: Yes
* LAMMPS: Yes, installed by the HZDR HPC team, it can be loaded by:

    .. code-block:: sh

        export MODULEPATH=$MODULEPATH:/trinity/shared/lmod/lmod/modulefiles/Core/experimental
        module load gcc/8.2.0
        module load openmpi/4.0.4
        module load python/3.8.0
        module load lammps/CG

* Quantum Espresso: Yes, installed by the HZDR HPC team, it can be loaded by:

    .. code-block:: sh

        export MODULEPATH=$MODULEPATH:/trinity/shared/lmod/lmod/modulefiles/Core/experimental
        module load gcc/10.2.0
        module load openmpi/2.1.2
        module load python/3.6.5
        module load qe/casus
