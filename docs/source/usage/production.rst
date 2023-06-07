MALA production inference runs
==============================

Optimal performance
*******************

When running MALA for large systems or extended periods of time, performance
becomes crucial. Most parts of MALA are either designed in a way to run as
efficient as possible by default. Enabling MPI and GPU accordingly naturally
boost performance by giving MALA access to more computational power. Currently,
only MPI **or** GPU acceleration can be used at the same time.

There are a few other options that can help with performance but need to be
handled with care to avoid numerical inaccuracies.

Gaussian descriptor based Ewald sum
------------------------------------

MALA by default employs Quantum ESPRESSO to evaluate the Ewald sum, i.e., the
ion-ion interaction part of the DFT energy. The implementation of this
algorithm in QE scales quadratically with the number of ions. Therefore MALA
also provides a custom, linearly scaling version of this algorithm through
the total energy module. This implementation is based on a representation of
atomic positions as Gaussians and can easily enabled:

      .. code-block:: python

            parameters.descriptors.use_gaussian_descriptors_energy_formula = True

It's numerical accuracy is governed by two parameters,
``parameters.descriptors.gaussian_descriptors_cutoff`` and
``parameters.descriptors.gaussian_descriptors_sigma``, the cutoff radius for
the Gaussian descriptors and the width of the Gaussian, respectively. MALA
automatically chooses reasonable default values that have been tested across
multiple systems. When dealing with a new system it may still be wise to
first test the automatically generated value against the default Quantum
ESPRESSO implementation before doing a full production run, to avoid
inaccuracies.
