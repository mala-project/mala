Advanced options
================

The following guide gives an overview over advanced capabilities of MALA,
such as performance improvements and more in-depth hyperparameter searches.
Parallelization and hardware acceleration will also be discussed. When using
the former, i.e., when computing across multiple CPUs, it is useful to
employ the MALA parallel-safe printing function for all output.

      .. code-block:: python

            import mala

            mala.printout("This message will only be printed once, even "
                           "if multiple CPUS are used")


.. toctree::
   :maxdepth: 1

   advanced_usage/trainingmodel
   advanced_usage/openpmd
   advanced_usage/descriptors
   advanced_usage/hyperparameters
   advanced_usage/predictions
