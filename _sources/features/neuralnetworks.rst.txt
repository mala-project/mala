Neural Networks
=================

Neural networks are powerful machine learning tools in principle capable of
approximating any function. In MALA, neural networks are built using PyTorch.
Hyperparameter optimization can be done using optuna and custom routines.

Data Handling
#############

MALA provides a ``DataHandler`` class that takes care of all the necessary data
operations. It loads, scales and performs inference on data, as well as
unit conversion. Raw simulation data has to be processed into data that can
be used with a ``DataHandler`` using the ``DataConverter`` class. In MALA,
data handling works on a per-snapshot basis; i.e. paths to individual snapshots are
added to the data handler via ``add_snapshot``.
After all snapshots of interests are given to
the data handler the necessary routines outlined above can be started with
``prepare_data``.
Large datasets can be lazily-loaded into the RAM, i.e. only a subset of the
entire data will be present in the RAM simultaneously.

Training and inference
######################

Neural networks are built using ``pytorch`` and the ``Network`` class. They
can be trained using the ``Trainer`` class and can be tested using the
``Tester`` class. Currently, only feed-forward neural-networks are implemented.

Hyperparameter optimization
###########################

Hyperparameter optimization can be done using optuna. Two experimental methods
are supported as well, orthogonal array tuning and neural architecture search
without training. In order to perform a hyperparameter optimization, a
``HyperOptOptuna`` object (for optuna) is created. Hyperparameters can be
added to the study with the ``add_hyperparameter`` function. Afterwards a
hyperparameter study can be executed.
Optuna can be used in a distributed fashion, using e.g. PostgreSQL.
