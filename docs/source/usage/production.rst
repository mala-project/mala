MALA production runs
====================

Optimal performance
*******************

When running MALA for large systems or extended periods of time, performance
becomes crucial. Most parts of MALA are either designed in a way to run as
efficient as possible by default. Enabling MPI and GPU accordingly naturally
boost performance by giving MALA access to more computational power.

There are however some settings that can help with performance but still
require some user intuition to be set up correctly. We are working on solutions
to automate these parts of the code as well, but for now, they still need to
be manually adjusted.

Gaussian descriptor based Ewald sum
------------------------------------

MALA by default employs Quantum ESPRESSO to evaluate the Ewald sum, i.e., the
ion-ion interaction part of the DFT energy. The implementation of this
algorithm in QE scales quadratically with the number of ions. Therefore MALA
also provides a custom, linearly scaling version of this algorithm through
the total energy module. This implementation can be used by setting three
parameters:

      .. code-block:: python

            parameters.descriptors.use_gaussian_descriptors_energy_formula = True
            parameters.descriptors.gaussian_descriptors_cutoff = ABC
            parameters.descriptors.gaussian_descriptors_sigma = XYZ

The first parameter, ``use_gaussian_descriptors_energy_formula`` simply
activates the usage of this algorithm. The second parameter
``gaussian_descriptors_cutoff`` determines, as the name would suggest, the
cutoff radius for the Gaussian descriptor calculation. It can safely be set to
the same value that was used as SNAP cutoff radius during model training.
The last parameter is the critical one. It determines the width of the
Gaussian used for descriptor construction. It is different for different
elements. Currently known values that will work well are:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Element
     - Sigma
   * - Beryllium
     - 0.3
   * - Aluminium
     - 0.135
