.. _production:

Using MALA in production
========================

MALA is aimed at providing ML-DFT models for large scale investigations.
Predictions at scale in principle work just like the predictions shown
in the basic guide. One simply has to set a few additional parameters to make
optimal use of the hardware at hand.

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
at the moment. Once MPI is activated, you can start the MPI aware python script
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

