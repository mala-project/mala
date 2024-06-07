.. _advanced training:

Improved training performance
==============================

MALA offers a options to make training available at large scales in reasonable
amounts of time. The general training workflow is the same as outlined in the
basic section; the advanced features can be activated by setting
the appropriate parameters in the ``Parameters`` object.

Using a GPU
***********

The simplest way to accelerate any NN training is to use a GPU for training.
Training NNs on GPUs is a well established industry practice and yields
huge speedups. MALA supports GPU training. You have to activate
GPU usage via

      .. code-block:: python

            parameters = mala.Parameters()
            parameters.use_gpu = True

Afterwards, the entire training will be performed on the GPU - given that
a GPU is available.

In cooperation with `Nvidia <https://www.nvidia.com/de-de/deep-learning-ai/solutions/machine-learning/>`_,
advanced GPU performance optimizations have been implemented into MALA.
Namely, you can enable the following options:

      .. code-block:: python

            parameters.use_gpu = False # True: Use GPU
            """
            Multiple workers allow for faster data processing, but require
            additional CPU/RAM power. A good setup is e.g. using 4 CPUs attached
            to one GPU and setting the num_workers to 4.
            """
            parameters.running.num_workers = 0 # set to e.g. 4
            """
            MALA supports a faster implementation of the TensorDataSet class
            from the torch library. Turning it on will drastically improve
            performance.
            """
            parameters.data.use_fast_tensor_data_set = False # True: Faster data loading
            """
            Likewise, using CUDA graphs improve performance by optimizing GPU
            usage. Be careful, this option is only availabe from CUDA 11.0 onwards.
            CUDA graphs will be most effective in cases that are latency-limited,
            e.g. small models with shorter epoch times.
            """
            parameters.running.use_graphs = False # True: Better GPU utilization
            """
            Using mixed precision can also improve performance, but only if
            the model is large enough.
            """
            parameters.running.use_mixed_precision = False # True: Improved performance for large models

These options aim at different performance bottlenecks in the training and have
been shown to speed-up training by up to a factor of 5 if used together,
while maintaining the same prediction accuracy.

Currently, these options are disabled by default as they are still being tested
extensively by the MALA team in production. **Yet, activating them is highly recommended!**

Advanced training metrics
****************************

When monitoring an NN training, one often checks the validation loss, which
is directly outputted by MALA. By default, this validation loss gives the
mean squared error between LDOS prediction and actual value. From a purely
ML point of view, this is fine; however, the correctness of the LDOS itself
does not hold much physical virtue. Thus, MALA implements physical validation
metrics to be accessed before and after the training routine.

Specifically, when setting

      .. code-block:: python

            parameters.running.after_before_training_metric = "band_energy"

the error in the band energy between actual and predicted LDOS will be
calculated and printed before and after network training (in meV/atom).
This is a much more intuitive metric to judge network performance, since
it is easily phyiscally interpretable. If the error is, e.g., 5 meV/atom, one
can expect the network to be reasonably accurate; values of, e.g., 100 meV/atom
hint at bad model performance. Of course, the final metric for the accuracy
should always be the results of the ``Tester`` class.

Please make sure to set the relevant LDOS parameters when using this property
via

      .. code-block:: python

            parameters.targets.ldos_gridsize = 11
            parameters.targets.ldos_gridspacing_ev = 2.5
            parameters.targets.ldos_gridoffset_ev = -5

Checkpointing a training run
****************************

NN training can take a long time, and on HPC systems, where they are usually
performed, there exist time limitations for calculations. Thus, it is often
necessary to checkpoint a training run and resume it at a later point.
MALA provides functionality for this, as shown in the example ``advanced/ex01_checkpoint_training.py``.
To use checkpointing, enable the feature in the ``Parameters`` object:

      .. code-block:: python

            parameters.running.checkpoints_each_epoch = 5
            parameters.running.checkpoint_name = "ex01_checkpoint"

Simply set an interval for checkpointing and a name for the checkpoint and
the training will automatically be checkpointed. Automatic resumption
from a checkpoint can trivially be implemented via

      .. code-block:: python

            if mala.Trainer.run_exists("ex01_checkpoint"):
                parameters, network, datahandler, trainer = \
                    mala.Trainer.load_run("ex01_checkpoint")
            else:
                parameters, network, datahandler, trainer = initial_setup()

Where ``initial_setup()`` encapsulates the training run setup.

Using lazy loading
******************

Lazy loading was already briefly mentioned during the testing of a network.
To recap, the idea of lazy loading is to incrementally load data into
memory so as to save on RAM usage in cases where large amounts of data are
involved. To use lazy loading, enable it by:

      .. code-block:: python

            parameters.data.use_lazy_loading = True


MALA lazy loading operates snapshot wise - that means if lazy loading is
enabled, one snapshot at a time is loaded into memory, processed, unloaded,
and the next one is selected. Thus, lazy loading *will* adversely
affect performance. One way to mitigate this is to use multiple CPUs to
load and prepare data, i.e., while one CPU is busy processing data/offloading
it to GPU, another CPU can already load the next snapshot into memory.
To use this so called "prefetching" feature, enable the corresponding
parameter via

      .. code-block:: python

            parameters.data.use_lazy_loading_prefetch = True

Please note that in order to use this feature, you have to assign enough
CPUs and memory to your calculation.

Apart from performance, there is an accuracy drawback when employing
lazy loading. It is well known that ML algorithms perform optimal when
individual training data points are accessed individually. This, however,
is not naively possible when using lazy loading - since data is not loaded
into memory completely at one point, data cannot be easily randomized.
This can impact accuracy very negatively for complicated data sets,
as briefly discussed in the MALA publication on
`temperature transferability of ML-DFT models <https://arxiv.org/abs/2306.06032>`_.

To circumvent this problem, MALA provides functionality to shuffle data from
multiple atomic snapshots in snapshot-like files, which can then be used
with lazy loading, guaranteeing randomized access to individual data points.
Currently, this method requires additional disk space, since the randomized
data sets have to be saved - in-memory implementations are currently developed.
To use the data shuffling (also shown in example
``advanced/ex02_shuffle_data.py``), you can use the ``DataShuffler`` class.

The syntax is very easy, you create a ``DataShufller`` object,
which provides the same ``add_snapshot`` functionalities as the ``DataHandler``
object, and shuffle the data once you have added all snapshots in question,
i.e.,

      .. code-block:: python

            parameters.data.shuffling_seed = 1234

            data_shuffler = mala.DataShuffler(parameters)
            data_shuffler.add_snapshot("Be_snapshot0.in.npy", data_path,
                                       "Be_snapshot0.out.npy", data_path)
            data_shuffler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                       "Be_snapshot1.out.npy", data_path)
            data_shuffler.shuffle_snapshots(complete_save_path="../",
                                            save_name="Be_shuffled*")

The seed ``parameters.data.shuffling_seed`` ensures reproducibility of data
sets. The ``shuffle_snapshots`` function has a path handling ability akin to
the ``DataConverter`` class. Further, via the ``number_of_shuffled_snapshots``
keyword, you can fine-tune the number of new snapshots being created.
By default, the same number of snapshots as had been provided will be created
(if possible).

Using tensorboard
******************

Training routines in MALA can be visualized via tensorboard, as also shown
in the file ``advanced/ex03_tensor_board``. Simply enable tensorboard
visualization prior to training via

      .. code-block:: python

            # 0: No visualizatuon, 1: loss and learning rate, 2: like 1,
            # but additionally weights and biases are saved
            parameters.running.visualisation = 1
            parameters.running.visualisation_dir = "mala_vis"

where ``visualisation_dir`` specifies some directory in which to save the
MALA visualization data. Afterwards, you can run the training without any
other modifications. Once training is finished (or during training, in case
you want to use tensorboard to monitor progress), you can launch tensorboard
via

      .. code-block:: bash

            tensorboard --logdir path_to_visualization

The full path for ``path_to_visualization`` can be accessed via
``trainer.full_visualization_path``.


Training in parallel
********************

If large models or large data sets are employed, training may be slow even
if a GPU is used. In this case, multiple GPUs can be employed with MALA
using the ``DistributedDataParallel`` (DDP) formalism of the ``torch`` library.
To use DDP, make sure you have `NCCL <https://developer.nvidia.com/nccl>`_
installed on your system.

To activate and use DDP in MALA, almost no modification of your training script
is necessary. Simply activate DDP in your ``Parameters`` object. Make sure to
also enable GPU, since parallel training is currently only supported on GPUs.

      .. code-block:: python

            parameters = mala.Parameters()
            parameters.use_gpu = True
            parameters.use_ddp = True

MALA is now set up for parallel training. DDP works across multiple compute
nodes on HPC infrastructure as well as on a single machine hosting multiple
GPUs. While essentially no modification of the python script is necessary, some
modifications for calling the python script may be necessary, to ensure
that DDP has all the information it needs for inter/intra-node communication.
This setup *may* differ across machines/clusters. During testing, the
following setup was confirmed to work on an HPC cluster using the
``slurm`` scheduler.

    .. code-block:: bash

        #SBATCH --nodes=NUMBER_OF_NODES
        #SBATCH --ntasks-per-node=NUMBER_OF_TASKS_PER_NODE
        #SBATCH --gres=gpu:NUMBER_OF_TASKS_PER_NODE
        # Add more arguments as needed
        ...

        # Load more modules as needed
        ...

        # This port can be arbitrarily chosen.
        # Given here is the torchrun default
        export MASTER_PORT=29500

        # Find out the host node.
        echo "NODELIST="${SLURM_NODELIST}
        master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
        export MASTER_ADDR=$master_addr
        echo "MASTER_ADDR="$MASTER_ADDR

        # Run using srun.
        srun -N NUMBER_OF_NODES -u bash -c '
        # Export additional per process variables
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=$SLURM_LOCALID
        export WORLD_SIZE=$SLURM_NTASKS

        python3 -u  training.py
        '

An overview of environment variables to be set can be found `in the official documentation <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`_.
A general tutorial on DDP itself can be found `here <https://pytorch.org/tutorials/beginner/ddp_series_theory.html>`_.


