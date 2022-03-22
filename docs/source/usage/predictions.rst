Predict electronic structures
=============================

Testing a model
***************

Once a model is trained, we need to assert its usefulness. For this, MALA
provides a ``Tester`` class. With it, data will be passed through a network
snapshot-wise, and we get access to actual and predicted LDOS, which can then
be further compared. See e.g. ``ex05_training_with_postprocessing``.

Making a prediction
********************

If we are content with the model performance, we can use the model
to make predictions. For this, MALA provides a ``Predictor`` class, that uses
a model to make a prediction about the electronic structure of an
atomic configuration in the form of an ``ase.Atoms`` object. Make sure to
always specify the parameters for descriptor calculation prior to this, i.e.


      .. code-block:: python

            parameters.running.inference_data_grid = [18, 18, 27]
            parameters.descriptors.descriptor_type = "SNAP"
            parameters.descriptors.twojmax = 10
            parameters.descriptors.rcutfac = 4.67637

elsewise the prediction will fail. See also ``ex12_run_predictions.py``.

Using Calculators
******************

The ``Predictor`` class can also be accessed through an ``ase`` calculator.
For this, use the ``mala.MALA`` class. See e.g. ``ex15_ase_calculator``.

.. warning:: Currently, the calculation of forces with this calculator is not officially supported.
