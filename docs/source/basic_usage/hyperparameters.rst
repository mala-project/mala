Basic hyperparameter optimization
=================================

With new data, it may be necessary to determine the hyperparameters we
assumed to be correct up until now manually. By default, MALA uses the
`optuna library <https://optuna.org/>`_ to tune hyperparameters.
:ref:`Advanced/experimental hyperparameter optimization strategies <advanced hyperparams>` are available as
well. This guide follows example ``ex04_hyperparameter_optimization``

In order to tune hyperparameters,
we first have to create a ``Parameters`` object, specify parameters,
create a ``DataHandler`` object and fill it with data. These steps are
essentially the same as the ones in the :doc:`training example <trainingmodel>`.

      .. code-block:: python

          parameters = mala.Parameters()
          parameters.data.input_rescaling_type = "feature-wise-standard"
          ...
          parameters.hyperparameters.n_trials = 20
          datahandler = mala.DataHandler(parameters)
          datahandler.add_snapshot(...)
          datahandler.add_snapshot(...)
          ...
          data_handler.prepare_data()

There are two noteworthy differences: Firstly, we do not have to specify
hyperparameters object when customizing the ``Parameters`` object that we
may want to tune later on; further, we have to specify the number of trials
via ``n_trials``. A *trial* is a candidate network/training strategy that is
tested by the hyperparameter optimization algorithm. Each hyperparameter
optimization *study* consists of multiple such trials, in which several
combinations of hyperparameters of interest are investigated and the best
one is identified.

The interface for adding hyperparameters to a study in MALA is

      .. code-block:: python

            hyperoptimizer = mala.HyperOpt(parameters, data_handler)
            hyperoptimizer.add_hyperparameter("categorical", "learning_rate",
                                              choices=[0.005, 0.01, 0.015])
            hyperoptimizer.add_hyperparameter(
                "categorical", "ff_neurons_layer_00", choices=[32, 64, 96])
            hyperoptimizer.add_hyperparameter(
                "categorical", "ff_neurons_layer_01", choices=[32, 64, 96])
            hyperoptimizer.add_hyperparameter("categorical", "layer_activation_00",
                                              choices=["ReLU", "Sigmoid", "LeakyReLU"])

Here, we have added the learning rate, number of neurons for two hidden NN
layers and the activation function in between to the hyperparameter
optimization. A reference list for potential hyperparameters and choices
is explained at the end of this section.

Once we have decided on hyperparameters, the actual hyperparameter optimization
can easily be accessed with

      .. code-block:: python

            hyperoptimizer.perform_study()
            hyperoptimizer.set_optimal_parameters()

The last command saves the determined, optimal hyperparameters to the
``Parameters`` object used in the script. The parameters can then easily
be saved to a ``.json`` file and loaded later, e.g., for training a new model.

      .. code-block:: python

            hyperoptimizer.perform_study()
            params = mala.Parameters.load_from_file(...)

List of hyperparameters
***********************

For in-depth description of how hyperparameter optimization works and an
extended explanation of parameters, please refer to the MALA publication
on `hyperparameter optimization <https://doi.org/10.1088/2632-2153/ac9956>`_.

MALA follows the optuna library in its nomenclature of hyperparameters. That
means, among other things, that each hyperparameter can either be

* ``"categorical"`` - a list of float values will be given as optimization space

* ``"float"`` - a lower and upper bound will be given as the optimization space, and the hyperparameter can be any real number in between

* ``"int"`` - a lower and upper bound will be given as the optimization space, and the hyperparameter can be any integer value in between

The following hyperparameters can be optimized, all of which correspond to
properties of the ``Parameters`` class:

.. list-table:: List of hyperparameters
   :widths: 10 5 10 10
   :header-rows: 1

   * - Name of the hyperparameter
     - Meaning
     - Linked parameter object
     - Possible choices
   * - ``"learning_rate"``
     - Learning rate of NN optimization (step size of gradient based optimizer)
     - ``running.learning_rate``
     - ``"float"``, ``"categorical"``
   * - ``"ff_multiple_layers_neurons"``
     - Has to be used in conjunction with ``"ff_multiple_layers_count"`` and is
       mutually exclusive with ``"ff_neurons_layer"``. Opti
     - ``network.layer_sizes``
     - ``"float"``, ``"categorical"``

