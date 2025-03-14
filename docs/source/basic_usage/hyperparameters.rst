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
   * - ``"ff_multiple_layers_neurons"`` /  ``"ff_multiple_layers_count"``
     - Always have to be used together and are
       mutually exclusive with ``"ff_neurons_layer"``. When using these options,
       the hyperparameter search will add multiple layers of the same size.
       ``"ff_multiple_layers_count"`` governs the number of layers added per
       trial, ``"ff_multiple_layers_neurons"`` the number of neurons per
       such layer.
     - ``network.layer_sizes``
     - ``"int"``, ``"categorical"``
   * - ``"ff_neurons_layer_XX"``
     - Number of neurons per layer. This is the primary tuning parameter
       to optimize the network architecture. One such parameter has to
       be added per potential NN layer, which is done by setting, e.g.,
       ``"ff_neurons_layer_00"``, ``"ff_neurons_layer_01"``, etc.;
       By including 0 in the list of choices, layers can be deactivted
       during the optimization.
     - ``network.layer_sizes``
     - ``"int"``, ``"categorical"``
   * - ``"optimizer"``
     - Optimization algorithm used during the NN optimization.
     - ``running.optimizer``
     - ``"categorical"``
   * - ``"mini_batch_size"``
     - Size of the mini batches used to calculate the gradient during
       the gradient-based NN optimization.
     - ``running.mini_batch_size``
     - ``"int"``, ``"categorical"``
   * - ``"early_stopping_epochs"``
     - If the validation loss does not decrease for this number of epochs,
       training is stopped.
     - ``running.early_stopping_epochs``
     - ``"int"``, ``"categorical"``
   * - ``"learning_rate_patience"``
     - If the validation loss does not decrease for this number of epochs,
       the learning rate is adjusted according to ``running.learning_rate_patience``.
     - ``running.learning_rate_patience``
     - ``"int"``, ``"categorical"``
   * - ``"learning_rate_decay"``
     - If the validation loss plateaus, then the learning rate is scaled by
       this factor. Should be smaller than zero.
     - ``running.learning_rate_decay``
     - ``"float"``, ``"categorical"``
   * - ``"layer_activation"``
     - Describes the activation functions used in the NN. Can either be a list
       used in the same fashion as ``"ff_neurons_layer_XX"``, i.e.,
       one hyperparameter per layer, or by only giving one hyperparameter,
       in which case all layers will use the same activation function.
     - ``network.layer_activation``
     - ``"categorical"``

