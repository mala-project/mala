Using horovod on slurm based HPC clusters
====================================================

As tested on hemera5 (HZDR)
---------------------------

The following suggestions are derived from tests on the slurm based cluster hemera5. They should be applicable to other slurm based clusters as well. You should always use all GPUs per node in a multinode setup.

If you want to run a MALA job on a slurm based machine, follow this outline as submit script::

    #!/bin/bash
    #SBATCH -N NUMBER_OF_NODES
    #SBATCH --ntasks-per-node=MAX_NUMBER_OF_GPUS_PER_NODE
    #SBATCH --gres=gpu:MAX_NUMBER_OF_GPUS_PER_NODE

    #.... other #SBATCH options ...

    #... module loading and such ...

    mpirun -np NUMBER_OF_NODES*MAX_NUMBER_OF_GPUS_PER_NODE -npernode MAX_NUMBER_OF_GPUS_PER_NODE -bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python3 your_mala_script.py