Training an ML-DFT model
========================

For a first glimpse into MALA, let's assume you already have input data
(i.e., bispectrum descriptors encoding the atomic structure on a real space
grid) and output data (the LDOS representing the electronic structure on the
real space grid) for a system. For now, we will be using the example data
provided in the `test data repository <https://github.com/mala-project/test-data>`_.

You will learn how to create your own data sets in :doc:`the following section <more_data>`.
This guide follows the example file ``basic/ex01_train_network.py`` and
``basic/ex02_test_network.py``.

Setting parameters
******************

The central object of MALA workflows is the ``mala.Parameters()`` object.
**All** options necessary to control a MALA workflow are accessible through
this object. To do this, it has several subobjects for its various tasks,
that we will get familiar with in this tutorial.

MALA provides reasonable choices for a lot of parameters of interest.
For a full list, please refer to the API reference - for now, we select a few
options to train a simple network with example data, namely

      .. code-block:: python

            parameters = mala.Parameters()

            parameters.data.input_rescaling_type = "feature-wise-standard"
            parameters.data.output_rescaling_type = "normal"

            parameters.network.layer_activations = ["ReLU"]

            parameters.running.max_number_epochs = 100
            parameters.running.mini_batch_size = 40
            parameters.running.learning_rate = 0.00001
            parameters.running.trainingtype = "Adam"
            parameters.verbosity = 1 # level of output; 1 is standard, 0 is low, 2 is debug.

Here, we can see that the ``Parameters`` object contains multiple
sub-objects dealing with the individual aspects of the workflow. In the first
two lines, which data scaling MALA should employ. Scaling data greatly
improves the performance of NN based ML models. Options are

* ``None``: No normalization is applied.

* ``standard``: Standardization (Scale to mean 0, standard deviation 1)

* ``normal``: Min-Max scaling (Scale to be in range 0...1)

* ``feature-wise-standard``: Row Standardization (Scale to mean 0, standard deviation 1)

* ``feature-wise-normal``: Row Min-Max scaling (Scale to be in range 0...1)

Here, we specify that MALA should standardize the input (=descriptors)
by feature (i.e., each entry of the vector separately on the grid) and
normalize the entire LDOS.

The third line tells MALA which activation function to use, and the last lines
specify the training routine employed by MALA.

For now, we will assume these values to be correct. Of course, for new
data sets, optimal values have to be determined via :doc:`hyperparameter optimization <hyperparameters>`.
Finally, it is useful to also save some information on how the LDOS and
bispectrum descriptors were calculated into the parameters object - this helps
at inference time, when this info is required. You will learn what these values
mean :doc:`data generation part <more_data>` of this guide, for now we
use the values consistent with the example data.

      .. code-block:: python

            parameters.targets.target_type = "LDOS"
            parameters.targets.ldos_gridsize = 11
            parameters.targets.ldos_gridspacing_ev = 2.5
            parameters.targets.ldos_gridoffset_ev = -5

            parameters.descriptors.descriptor_type = "Bispectrum"
            parameters.descriptors.bispectrum_twojmax = 10
            parameters.descriptors.bispectrum_cutoff = 4.67637

Adding training data
********************

As with any ML library, MALA is a data-driven framework. So before we can
train a model, we need to add data. The central object to manage data for any
MALA workflow is the ``DataHandler`` class.

MALA manages data "per snapshot". One snapshot is one atomic configuration,
for which volumetric input and output data has been calculated. Data has to
be added to the ``DataHandler`` object per snapshot, pointing to the
where the volumetric data files are saved on disk. This is done via

      .. code-block:: python

            data_handler = mala.DataHandler(parameters)
            data_handler.add_snapshot("Be_snapshot0.in.npy", data_path,
                                      "Be_snapshot0.out.npy", data_path, "tr")
            data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                      "Be_snapshot1.out.npy", data_path, "va")

The ``"tr"`` and ``"va"`` flag signal that the respective snapshots are added as
training and validation data, respectively. Training data is data the model
is directly tuned on; validation data is data used to verify the model
performance during the run time and make sure that no overfitting occurs.
After data has been added to the ``DataHandler``, it has to be actually loaded
and scaled via

      .. code-block:: python

            data_handler.prepare_data()

The ``DataHandler`` object can now be used for Machine learning.

Building and training a model
*****************************

MALA uses neural networks (NNs) as a backbone for the ML-DFT models. To
construct those, we have to specify the number of neurons. This is also done
via the ``Parameters`` object. In principle, we can specify the layer sizes
whenever we want, however, it makes sense to do this *after* the data has been
loaded, because then it is easier to make sure that the dimensions of the
layers agree. To build a NN, we specify

      .. code-block:: python

            parameters.network.layer_sizes = [data_handler.input_dimension,
                                              100,
                                              data_handler.output_dimension]
            network = mala.Network(parameters)


Now, we can easily train this network with the parameters specified above
by doing

      .. code-block:: python

            trainer = mala.Trainer(parameters, network, data_handler)
            trainer.train_network()

Afterwards, we want to save this model for future use. MALA saves models
in a ``*.zip`` format. Within each model archive, information like scaling
coefficients, the model weights itself, etc. are stored in one place where MALA
can easily access it. Additionally, it makes sense to provide MALA with a
sample calculation output (from the simulations used to gather the training
data), so that critical parameters like simulation temperature, grid
coarseness, etc., are available at inference time. By

      .. code-block:: python

            additional_calculation_data = os.path.join(data_path, "Be_snapshot0.out")
            trainer.save_run("be_model",
                             additional_calculation_data=additional_calculation_data)

This information is set and the resulting model is saved. It is now ready to
be used.

Testing a model
***************

Before using a model in production, it is wise to test its performance. To that
end, MALA provides a ``Tester`` class, that allows users to load a model,
give it some data unseen during training, and verify the models performance
on that data.

This verification is done by selecting observables of interest (e.g., the band
energy, total energy or number of electrons) and comparing ML-DFT predictions
with the ground truth. To instantiate a ``Tester`` object, call

      .. code-block:: python

            parameters, network, data_handler, tester = mala.Tester.load_run("be_model")

There are a few useful options we should set when testing a network.
Firstly, we need to specify which observables to test. Secondly, we have to
decide if we want the resulting accuracy measures per each individual snapshot
(``"list"``) or as an average across all snapshots (``"mae"``).
Finally, it is useful to enable lazy-loading. Lazy-loading is a feature that
incrementally loads data into memory. It is necessary when operating on large
amounts of data; its usage in the training routine is further discussed in
:ref:`the advanced training section <advanced training>`.
For testing a model, it is prudent to enable, since a lot of data may
be involved. The accompanying syntax for these three options is

      .. code-block:: python

            tester.observables_to_test = ["band_energy", "number_of_electrons"]
            tester.output_format = "list"
            parameters.data.use_lazy_loading = True

Afterwards, new data can be added just as shown above, now with the data
function being ``"te"`` for testing data. Once this is done, testing can
be done via

      .. code-block:: python

            results = tester.test_all_snapshots()

Resulting in a dictionary, which can either be saved into a ``.csv`` file or
directly processed.
