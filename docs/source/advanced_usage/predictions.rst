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

Where you have to specify a list with three entries ``[x,y,z]``. As matter
of principle, stretching simulation cells in either direction should be
reflected by the grid.

Likewise, you can adjust the inference temperature via

      .. code-block:: python

            # Predictor class
            predictor.predict_for_atoms(atoms, temperature=...)
            # ASE calculator
            calculator.data_handler.target_calculator.temperature = ...


Predictions on GPU
*******************

MALA predictions can be run entirely on a GPU. For the NN part of the workflow,
this seems like a trivial statement, but the GPU acceleration extends to
descriptor calculation and total energy evaluation. By enabling GPU support
with

      .. code-block:: python

            parameters.use_gpu = True

prior to an ASE calculator calculation or usage of the ``Predictor`` class,
all computationally heavy parts of the MALA inference, will be offloaded
to the GPU.

Please note that this requires LAMMPS to be installed with GPU, i.e., Kokkos
support. A current limitation of this implementation is that only a *single*
GPU can be used for inference. This puts an upper limit on the number of atoms
which can be simulated, depending on the hardware you have access to.
Usual numbers observed by MALA team put this limit at a few thousand atoms, for
which the electronic structure can be predicted in 1-2 minutes. Currently,
multi-GPU inference is being implemented.

Parallel predictions on CPUs
****************************

Since GPU usage is currently limited to one GPU at a time, predictions
for ten- to hundreds of thousands of atoms rely on the usage of a large number
of CPUs. Just like with GPU acceleration, nothing about the general inference
workflow has to be changed. Simply enable MPI usage in MALA

      .. code-block:: python

            parameters.use_mpi = True

Please be aware that GPU and MPI usage are mutually exclusive for inference
at the moment. Once MPI is activated, you can start the MPI aware Python script
with a large number of CPUs to simulate materials at large length scales.

By default, MALA can only operate with a number of CPUs by which the
z-dimension of the inference grid can be evenly divided, since the Quantum
ESPRESSO backend of MALA by default only divides data along the z-dimension.
If you, e.g., have an inference grid of ``[200,200,200]`` points, you can use
a maximum of 200 CPUs. Using, e.g., 224 CPUs will lead to an error.

Parallelization can further be made more efficient by also enabling splitting
in the y-dimension. This is done by setting the parameter

      .. code-block:: python

            parameters.descriptors.use_y_splitting = ysplit

to an integer value ``ysplit`` (default: 0). If ``ysplit`` is not zero,
each z-plane will be divided ``ysplit`` times for the parallelization.
If you, e.g., have an inference grid of ``[200,200,200]``, you could use
400 CPUs and ``ysplit`` of 2. Then, the grid will be sliced into 200 z-planes,
and each z-plane will be sliced twice, allowing even faster inference.

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

