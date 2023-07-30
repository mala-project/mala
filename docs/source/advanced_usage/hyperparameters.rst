.. _advanced hyperparams:

Improved hyperparameter optimization
=====================================

Checkpointing a hyperparameter search
*************************************

Just like a regular NN training, a hyperparameter search may be checkpointed
and resumed at a later point. An example is shown in the file
``advanced/ex05_checkpoint_hyperparameter_optimization.py``.
The syntax/interface is very similar as to the NN training checkpointing
system, i.e.:

      .. code-block:: python

            import mala

            parameters = mala.Parameters()

            parameters.hyperparameters.checkpoints_each_trial = 5
            parameters.hyperparameters.checkpoint_name = "ex05_checkpoint"

will enable hyperparameter checkpointing after 5 hyperparameter trials.
The checkpointing-restarting is then enabled via

      .. code-block:: python

            if mala.Trainer.run_exists("ex05_checkpoint"):
                parameters, network, datahandler, trainer = \
                    mala.Trainer.load_run("ex05_checkpoint")
                printout("Starting resumed training.")
            else:
                parameters, network, datahandler, trainer = initial_setup()
                printout("Starting original training.")

Parallelizing a hyperparameter search
**************************************

Hyperparameter searches can be very computationally demanding, since many
NN trainings have to be performed in order to determine the optimal set
of parameters. One way to accelerate the process is through a
*parallel*, distributed hyperparameter search. Here, we run the hyperparameter
search on N CPUs/GPUs - each processing one specific trial. The results
are collected in a central data base, which is accessed by each of the N
ranks individually upon starting new trials. This is not a MALA feature itself,
but rather a feature of the optuna library, that MALA implements and
gives easy access to.

Database handling is done via SQL. Several SQL frameworks exists, and
optuna/MALA support PostgreSQL, MySQL and SQLite. The latter is good for local
debugging, but should not be used at scale. An SQLite example for distributed
hyperparameter optimization can be found in the file
``advanced/ex06_distributed_hyperparameter_optimization.py``.

Distributed hyperparameter optimization works the same as regular
hyperparameter optimization; parameter, data and optimizer setup do not have
to be altered. You have to specify the data base (and a name for the
study, since multiple studies may be saved in the same data base) to be used,
e.g.,

      .. code-block:: python

            parameters.hyperparameters.study_name = "my_study"
            # SQLite example
            parameters.hyperparameters.rdb_storage = 'sqlite:///DATABASENAME'
            # PostgreSQL example
            parameters.hyperparameters.rdb_storage = "postgresql://SERVERNAME/DATABASENAME"

For more information on `PostgreSQL <https://www.postgresql.org/>`_ and
`SQLite <https://www.sqlite.org/index.html>`_, please visit the respective
websites.

Once this parameter is set, the MALA-optuna interface will automatically use
this database to store and initialize all trials. If you then execute the
hyperparameter optimization Python script *multiple* times, the hyperparameter
optimization will be parallel - no further changes to the code are necessary.

On an HPC cluster, it may be prudent to launch the script multiple times
with a single call, e.g.

      .. code-block:: bash

            mpirun -np 12 python3 -u hyperopt.py

since there a limitations on how many jobs can be launched individually.
If you have to do this, MALA provides the helpful option of disentangling these
instances via

      .. code-block:: python

            parameters.optuna_singlenode_setup(wait_time=10)

This command makes sure that the individual instances of the optuna run
are started with ``wait_time`` time interval in between (to avoid race
conditions when accessing the same data base) and further only use the data
base, not MPI, for communication.

If you do distributed hyperparameter optimization, another useful option
is

      .. code-block:: python

            parameters.hyperparameters.number_training_per_trial = 3

This option tells optuna to run each NN trial training
``number_training_per_trial`` times; instead of the accuracy of only one
trial, the average accuracy (plus the standard deviation) are reported at the
end of each trial. Doing so massively increases robustness of the
hyperparameter optimization, since it eliminates models that perform well
by chance (e.g., because they have been randomly initialized to be accurate
by chance). This option is especially useful if used in conjunction with
a physical validation metric such as

      .. code-block:: python

            parameters.running.after_before_training_metric = "band_energy"

Advanced optimization algorithms
********************************

As discussed in the MALA publication on
`hyperparameter optimization <https://doi.org/10.1088/2632-2153/ac9956>`_,
advanced hyperparameter optimization strategies have been evaluated for
ML-DFT models with MALA. Namely

* NASWOT (Neural architecture search without training):
  A training-free hyperparameter optimization technique. It works by
  correlating the capability of a network to distinguish between data points
  at NN initialization with performance after training.
* OAT (Orthogonal array tuning):
  This technique requires network training, but constructs an optimal set
  of trials based on orthogonal arrays (a concept from optimal design theory)
  from which to extract a maximum of information with a limited number of
  training overhead.

Both methods can easily be enabled without changing the familiar hyperparameter
optimization workflow, as shown in the file
``advanced/ex07_advanced_hyperparameter_optimization``.

These optimization algorithms are activated via the ``Parameters`` object:

      .. code-block:: python

            # Use NASWOT
            parameters.hyperparameters.hyper_opt_method = "naswot"
            # Use OAT
            parameters.hyperparameters.hyper_opt_method = "oat"

Both techniques are fully compatible with other MALA capabilities, with
a few exceptions:

* NASWOT: Can only be used with hyperparameters related to network architecture
  (layer sizes, activation functions, etc.); training related hyperparameters
  will be ignored, and a warning to this effect will be printed. Only
  ``"categorical"`` hyperparameters are supported. Can be run in parallel by
  setting ``parameters.use_mpi=True``.
* OAT: Can currently not be run in parallel. Only
  ``"categorical"`` hyperparameters are supported.

For more details on the mathematical background of these methods, please refer
to the aforementioned publication.



