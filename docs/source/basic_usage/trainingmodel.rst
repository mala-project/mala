Training an ML-DFT model
========================

For a first glimpse into MALA, let's assume you already have input data
(i.e., bispectrum descriptors encoding the atomic structure on a real space
grid) and output data (the LDOS representing the electronic structure on the
real space grid) for a system. For now, we will be using the example data
provided in the `test data repository <https://github.com/mala-project/test-data>`_.

You will learn how to create your own data sets in :doc:`the following section <more_data>`.
This guide follows the example file ``basic/ex01_train_network.py``.

Setting parameters
******************

The central object of MALA workflows is the ``mala.Parameters()`` object.
**All** options necessary to control a MALA workflow are accessible through
this object. To do this, it has several subobjects for its various tasks,
that we will get familiar with in the following.

For now, we select a few options to train a simple network with example data,
namely

      .. code-block:: python

            parameters = mala.Parameters()

            parameters.data.input_rescaling_type = "feature-wise-standard"
            parameters.data.output_rescaling_type = "normal"

            parameters.network.layer_activations = ["ReLU"]

            parameters.running.max_number_epochs = 100
            parameters.running.mini_batch_size = 40
            parameters.running.learning_rate = 0.00001
            parameters.running.trainingtype = "Adam"

Here, we can see that the ``mala.Parameters`` object contains multiple
sub-objects dealing with the individual aspects of the workflow. In the first
two lines, we specify that MALA should standardize the input (=descriptors)
by feature (i.e., each entry of the vector separately on the grid) and
normalize the entire LDOS. The third line tells MALA which activation
function to use, and the last lines specify the training routine employed
by MALA.

For now, we will assume these values to be correct. Of course, for new
data sets, optimal values have to be determined via :doc:`hyperparameter optimization <hyperparameters>`.
Finally, it is useful to also save some information on how the LDOS and
bispectrum descriptors were calculated into the parameters object - this helps
at inference time, when this info is required. You will learn what these values
mean :doc:`data acquisition part <more_data>` of this guide, for now we simply
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

