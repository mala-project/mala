.. _production:

Using MALA in production
========================

MALA is aimed at providing ML-DFT models for large scale investigations.
Predictions at scale in principle work just like the predictions shown
in the basic guide. One has to set a few additional parameters to make
optimal use of the hardware at hand.

As a general remark please be reminded that if you have not used LAMMPS
for your first steps in MALA, and instead used the python-based descriptor
calculation methods, we highly advise switching to LAMMPS for advanced/more
involved examples (see  :ref:`installation instructions for LAMMPS <lammpsinstallation>`).

MALA ML-DFT models can be used for predictions at system sizes and temperatures
larger resp. different from the ones they were trained on. If you want to make
a prediction at a larger length scale then the ML-DFT model was trained on,
MALA will automatically determine a suitable realspace grid for the prediction.
You can manually specify the inference grid if you wish via

      .. code-block:: python

            # Predictor class
            parameters.running.inference_data_grid = ...
            # ASE calculator
            calculator.mala_parameters.running.inference_data_grid = ...

Here you have to specify a list with three entries ``[x,y,z]``. As matter
of principle, stretching simulation cells in either direction should be
reflected by the grid.

Likewise, you can adjust the inference temperature via

      .. code-block:: python

            # Predictor class
            predictor.predict_for_atoms(atoms, temperature=...)
            # ASE calculator
            calculator.data_handler.target_calculator.temperature = ...


.. _production_gpu:

Predictions on GPUs
*******************

MALA predictions can be run entirely on a GPU. For the NN part of the workflow,
this seems like a trivial statement, but the GPU acceleration extends to
descriptor calculation and total energy evaluation. By enabling GPU support
with

      .. code-block:: python

            parameters.use_gpu = True

prior to an ASE calculator calculation or usage of the ``Predictor`` class,
all computationally heavy parts of the MALA inference, will be offloaded
to the GPU. Please note that this requires LAMMPS to be installed with GPU, i.e., Kokkos
support. Multiple GPUs can be used during inference by first enabling
parallelization via

      .. code-block:: python

            parameters.use_mpi = True

and then invoking the MALA instance through ``mpirun``, ``srun`` or whichever
MPI wrapper is used on your machine. Details on parallelization
are provided :ref:`below <production_parallel>`.

.. note::

    To use GPU acceleration for total energy calculation, an additional
    setting has to be used.

Currently, there is no direct GPU acceleration for the total energy
calculation. For smaller calculations, this is unproblematic, but it can become
an issue for systems of even moderate size. To alleviate this problem, MALA
provides an optimized total energy calculation routine which utilizes a
Gaussian representation of atomic positions. In this algorithm, most of the
computational overhead of the total energy calculation is offloaded to the
computation of this Gaussian representation. This calculation is realized via
LAMMPS and can therefore be GPU accelerated (parallelized) in the same fashion
as the bispectrum descriptor calculation. Simply activate this option via

    .. code-block:: python

        parameters.descriptors.use_atomic_density_energy_formula = True

The Gaussian representation algorithm is describe in
the publication `Predicting electronic structures at any length scale with machine learning <doi.org/10.1038/s41524-023-01070-z>`_.

.. _production_parallel:

Parallel predictions
********************

MALA predictions may be run on a large number of processing units, either
CPU or GPU. To do so, simply enable MPI usage in MALA

      .. code-block:: python

            parameters.use_mpi = True

Once MPI is activated, you can start the MPI aware Python script using
``mpirun``, ``srun`` or whichever MPI wrapper is used on your machine.

By default, MALA can only operate with a number of processes by which the
z-dimension of the inference grid can be evenly divided, since the Quantum
ESPRESSO backend of MALA by default only divides data along the z-dimension.
If you, e.g., have an inference grid of ``[200,200,200]`` points, you can use
a maximum of 200 ranks. Using, e.g., 224 CPUs will lead to an error.

Parallelization can further be made more efficient by also enabling splitting
in the y-dimension. This is done by setting the parameter

      .. code-block:: python

            parameters.descriptors.use_y_splitting = ysplit

to an integer value ``ysplit`` (default: 0). If ``ysplit`` is not zero,
each z-plane will be divided ``ysplit`` times for the parallelization.
If you, e.g., have an inference grid of ``[200,200,200]``, you could use
400 processes and ``ysplit`` of 2. Then, the grid will be sliced into 200
z-planes, and each z-plane will be sliced twice, allowing even faster
inference.

Visualizing observables
************************

MALA also provides useful functions to visualize observables, as shown in
the file ``advanced/ex08_visualize_observables``. To calculate observables
for analysis and visualization, you need an LDOS calculator object.
If you perform ML-DFT inference, you will get this object from the
``Predictor`` resp. ASE calculator object, but it can also be created by
itself, as shown in the mentioned example file.

Having obtained an LDOS calculator object, you can access several observables
of interest for visualization via

      .. code-block:: python

            # The DOS can be visualized on the correct energy grid.
            density_of_states = ldos_calculator.density_of_states
            energy_grid = ldos_calculator.energy_grid

            # The density can be saved into a .cube file for visualization with standard
            # electronic structure visualization software.
            density_calculator = mala.Density.from_ldos_calculator(ldos_calculator)
            density_calculator.write_to_cube("Be_density.cube")

            # The radial distribution function can be visualized on discretized radii.
            rdf, radii = ldos_calculator.\
                radial_distribution_function_from_atoms(ldos_calculator.atoms,
                                                        number_of_bins=500)

            # The static structure factor can be visualized on a discretized k-grid.
            static_structure, kpoints = ldos_calculator.\
                static_structure_factor_from_atoms(ldos_calculator.atoms,
                                                   number_of_bins=500, kMax=12)

With the exception of the electronic density, which is saved into the ``.cube``
format for visualization with regular electronic structure visualization
software, all of these observables can be plotted with Python based
visualization libraries such as ``matplotlib``.
