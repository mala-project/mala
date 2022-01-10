
# Modules for MLMM

module purge

module load sems-env
#module load kokkos-env
module load pyomo-env

module load sems-gcc/7.2.0
module load sems-openmpi/4.0.2
module load sems-cuda/10.1
module load python37/3.7.3

export HOROVOD_CUDA_INCLUDE=/projects/sems/install/rhel7-x86_64/sems/compiler/cuda/10.1/base/include
export HOROVOD_CUDA_LIB=/projects/sems/install/rhel7-x86_64/sems/compiler/cuda/10.1/base/lib64


module list

