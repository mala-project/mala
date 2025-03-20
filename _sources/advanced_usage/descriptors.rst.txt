.. _tuning descriptors:

Advanced descriptor options
===========================

As a general remark please be reminded that if you have not used LAMMPS
for your first steps in MALA, and instead used the python-based descriptor
calculation methods, we highly advise switching to LAMMPS for advanced/more
involved examples (see  :ref:`installation instructions for LAMMPS <lammpsinstallation>`).

Tuning descriptors
******************

The data conversion shown in the basic MALA guide is straightforward from
an interface point of view, however it is non-trivial to determine the
correct hyperparameters for the bispectrum descriptors. Ideally, cutoff radius
and dimensionality of the descriptors should be chosen such that atomic
environments around each point in space are accurately represented.

Some physical intuition factors into this process, as detailed in
the MALA publication on `hyperparameter optimization <https://doi.org/10.1088/2632-2153/ac9956>`_.
There, it is discussed that too large cutoff radii lead to descriptors
that are too similar for different points in space, complicating the
ML training, while too small radii lead to descriptors not containing
enough physical information, and do not yield physically accurate models.
Likewise, the dimensionality of the expansion of the atomic density
needs to match the amount of information one wants to encode.

In the publication mentioned above, a method has been devised based on these
principles, the so called average cosine similarity distance (ACSD) analysis.
Based on cosine similarities between descriptor and target vectors for
distinct points, optimal descriptor hyperparameters can be determined; for
more detail, please refer to the publication.

Within MALA, this analysis is available as a special hyperparameter
optimization routine, as showcased in the example ``advanced/ex04_acsd.py``.
The syntax for this analysis is similar to the regular hyperparameter
optimization interface. First, set up an optimizer via

      .. code-block:: python

            import mala

            parameters = mala.Parameters()

            # Specify the details of the ACSD analysis.
            parameters.descriptors.acsd_points = 100
            hyperoptimizer = mala.ACSDAnalyzer(parameters)

The ``acsd_points`` parameter determines how many points are compared during
the ACSD analysis; the more points you select, the longer the analysis
takes. If you set this value to 100, 100 points are compared with 100 different
points each, yielding a point cloud of 10.000 points for analysis. For most
purposes, this should be enough.

Afterwards, specify ranges for the bispectrum hyperparameters over which
to search for the optimal set of values via

      .. code-block:: python

            hyperoptimizer.add_hyperparameter("bispectrum_twojmax", [2, 4])
            hyperoptimizer.add_hyperparameter("bispectrum_cutoff", [1.0, 2.0])

These two are the only hyperparameters needed for the bispectrum descriptors.
Choose the lists according to the demands of your system; a good starting
point is a coarse search over cutoff radii, and ``bispectrum_twojmax``
values of 2 to a maximum of 10.

Afterwards, you have to add data to the hyperparameter optimization. This
is similar to the regular hyperparameter optimization workflow, with two
import distinctions. Firstly, only add preprocessed data for the LDOS; for
the descriptors, add a raw calculation output, from which the descriptors
can be computed. Secondly, the ``add_snapshot`` function is provided directly
via the hyperparameter optimizer. No ``DataHandler`` instance is needed.
An example would be this:

      .. code-block:: python

            hyperoptimizer.add_snapshot("espresso-out", os.path.join(data_path, "Be_snapshot1.out"),
                                        "numpy", os.path.join(data_path, "Be_snapshot1.out.npy"),
                                        target_units="1/(Ry*Bohr^3)")
            hyperoptimizer.add_snapshot("espresso-out", os.path.join(data_path, "Be_snapshot2.out"),
                                        "numpy", os.path.join(data_path, "Be_snapshot2.out.npy"),
                                        target_units="1/(Ry*Bohr^3)")

Once this is done, you can start the optimization via

      .. code-block:: python

            hyperoptimizer.perform_study(return_plotting=False)
            hyperoptimizer.set_optimal_parameters()

If ``return_plotting`` is set to ``True``, relevant plotting data for the
analysis are returned. This is useful for exploratory searches.

Since the ACSD re-calculates the bispectrum descriptors for each combination
of hyperparameters, it is useful to use parallel descriptor calculation.
To do so, you can enable the `MPI <https://www.mpi-forum.org/>`_ capabilites
of MALA/LAMMPS. Once enabled, multiple CPUs can be used in parallel to
calculate descriptors. Enabling MPI in MALA can easily be done via

      .. code-block:: python

            parameters.use_mpi = True

If you use MPI, multiple CPUs need to be allocated to the MALA computation.

Parallel data conversion
*************************

Parallelization may also generally be used for data conversion via the
``DataConverter`` class. Just enable the MPI function in MALA via

      .. code-block:: python

            parameters.use_mpi = True

prior to using the ``DataConverter`` class. Then, all processing will
be done in parallel - both the descriptor calculation as well as the LDOS
parsing.

ACE Descriptors
******************

.. note::

    To use ACE descriptors with MALA, you need to install LAMMPS from source
    using the ACE descriptor development branch, since the ACE descriptors
    are not yet part of the descriptor calculation code the MALA team has
    integrated into mainline LAMMPS. You can find the code here:
    https://github.com/jmgoff/lammps_compute_PACE/tree/mala-ace-grid.

Recently, and as described in the
`MALA technical paper <https://arxiv.org/abs/2411.19617>`_ ACE descriptors
have been implemented as an alternative to bispectrum descriptors. They
follow the Atomic Cluster Expansion (ACE) formalism, introduced by
the `eponymous publication <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104>`_
by Ralf Drautz. ACE descriptors hold the promise of being more descriptive and
accurate than bispectrum descriptors and are currently being investigated by
the MALA team. MALA already implements most functionalities of bispectrum
descriptors for ACE descriptors. You can use them in the same fashion as
the bispectrum descriptors, with the only difference being the hyperparameters
you need to set.

Specifically, by replacing all bispectrum hyperparameters in your script
with code such as this

        .. code-block:: python

            parameters.descriptors.descriptor_type = "ACE"
            parameters.descriptors.ace_cutoff = 5.8
            parameters.descriptors.ace_included_expansion_ranks = [1, 2, 3]
            parameters.descriptors.ace_maximum_l_per_rank = [0, 1, 1]
            parameters.descriptors.ace_maximum_n_per_rank = [1, 1, 1]
            parameters.descriptors.ace_minimum_l_per_rank = [0, 0, 0]

ACE descriptors will be used in your processing/training/testing scripts.
ACE_DOCS_MISSING: Describe what the parameters mean/how to best tune them.

A known current limitation is that ACE descriptors can only be run on CPU.
A GPU version is currently being developed.
