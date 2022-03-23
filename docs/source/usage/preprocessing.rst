Data generation and preprocessing
==================================

Data generation
###############

Data generation for MALA is done by electronic structure calculations using
appropriate simulation software. The raw outputs of such calculations
are atomic positions and the LDOS, although often as separate cube files.
Currently, only QuantumESPRESSO has been tested. More specifically, the MALA
team maintains a special version of QuantumESPRESSO. It is both used as
interface in MALA for the calculation of total energies, as well as for the
DFT calculations. It contains a modification which allows flexible LDOS
sampling, needed for this LDOS based workflow. It can be obtained at:

https://gitlab.com/casus/q-e/-/tree/tem_original_development

Data Conversion
###############

MALA can be used to process raw data into ready-to-use data fro the surrogate models
For this, the ``DataConverter`` class can be used; see example ``ex02_preprocess_data``.

Using input and output data
###########################

MALA views divides provided data into two categories: ``Descriptor`` objects
and ``Target`` objects.

Descriptors
***********

Descriptors give, per grid-point, information about the
local environment around that grid-point. In most cases, users will not have
to interact with a ``Descriptor`` object. Yet, these objects can be created
by themselves if needed.

      .. code-block:: python

            import mala

            parameters = mala.Parameters()
            parameters.descriptors.descriptor_type = 'SNAP'

            # Creates a SNAP object via interface
            snap = mala.Descriptor(parameters)

            # Creates a SNAP object directly
            snap = mala.SNAP(parameters)

            # Use the SNAP object to calculate descriptors from QE calculation.
            snap_descriptors = snap.calculate_from_qe_out(...)


Targets
*******

``Target`` objects hold information about the electronic structure of a material.
In comparison to the ``Descriptor`` objects these objects are more common
in workflows, as they can be used for a variety of physical calculations.
They can also be used to parse data from ab-initio simulations
E.g. a script like this:

      .. code-block:: python

            import mala

            parameters = mala.Parameters()
            parameters.targets.descriptor_type = 'LDOS'

            # Creates a LDOS object via interface
            ldos = mala.Target(parameters)

            # Creates a LDOS object directly
            ldos = mala.LDOS(parameters)

            # Use the LDOS object to calculate the band energy from LDOS data.
            band_energy = ldos.read_from_cube(...)


Data Scaling
############

An additional step of preprocessing is scaling the data before a model is
trained. This is done automatically in the ``DataHandler`` class, using the
methods requested via the ``mala.parameters.data.input_rescaling_type`` and
``mala.parameters.data.output_rescaling_type`` keywords. Currently supported here
are:

* ``None``: No normalization is applied.

* ``standard``: Standardization (Scale to mean 0, standard deviation 1)

* ``normal``: Min-Max scaling (Scale to be in range 0...1)

* ``feature-wise-standard``: Row Standardization (Scale to mean 0, standard deviation 1)

* ``feature-wise-normal``: Row Min-Max scaling (Scale to be in range 0...1)

Internally, the ``DataScaler`` class is used. The objects of this class
can be saved and loaded later for e.g. inference or to minimize calculation
time for multiple ML experiments using the same set of data.
Data scaling will always be done using the training data only.
