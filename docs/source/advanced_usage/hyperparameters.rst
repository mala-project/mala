.. _advanced hyperparams:

Improved hyperparameter optimization
=====================================

Checkpointing a hyperparameter search
*************************************

Just like a regular NN training, a hyperparameter search may be checkpointed
and resumed at a later point. An example is shown in the file
``advanced/ex04_checkpoint_hyperopt.py``.
The syntax/interface is very similar as to the NN training checkpointing
system, i.e.:

      .. code-block:: python

            parameters.hyperparameters.checkpoints_each_trial = 5
            parameters.hyperparameters.checkpoint_name = "ex04_checkpoint"

will enable hyperparameter checkpointing after 5 hyperparameter trials.
The checkpointing-restarting is then enabled via

      .. code-block:: python

            if mala.Trainer.run_exists("ex01_checkpoint"):
                parameters, network, datahandler, trainer = \
                    mala.Trainer.load_run("ex01_checkpoint")
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
but rather a feature of the optuna library, that MALA simply implements and
gives easy access to.

Database handling is done via SQL. Several SQL frameworks exists, and
optuna/MALA support PostgreSQL, MySQL and SQLite. The latter is good for local
debugging, but should not be used at scale. An SQLite example for distributed
hyperparameter optimization can be found in the file ``advanced/ex
