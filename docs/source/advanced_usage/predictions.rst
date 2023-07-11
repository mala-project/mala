.. _production:

Using MALA in production
========================

MALA is aimed at providing ML-DFT models for large scale investigations.
Predictions at scale in principle work just like the predictions shown
in the basic guide. One simply has to set a few additional parameters to make
optimal use of the hardware at hand.

Predictions on GPU
*******************

MALA predictions can be run entirely on a GPU. For the NN part of the workflow,
this seems like a trivial statement, but the GPU acceleration extends to
descriptor calculation and total energy evaluation. By enabling GPU support
with

      .. code-block:: python

            import mala

            parameters = mala.Parameters()
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
with a large number of CPUs to simulate materials with
