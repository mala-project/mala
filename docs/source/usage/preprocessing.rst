Data generation and preprocessing
==================================

Data generation
###############

Data generation for MALA is done by electronic structure calculations using
appropriate simulation software. The raw outputs of such calculations
are atomic positions and the LDOS, the latter usually as multiple individual
.cube files.

Currently, only QuantumESPRESSO has been tested. Starting with version 7.2,
any version of QuantumESPRESSO can be used to create data for MALA. In order
to do so

1. Perform a regular DFT calculation using ``pw.x``
2. Calculate the LDOS using ``pp.x``

Make sure to use enough k-points in the DFT calculation (LDOS sampling
requires denser k-grids then regular DFT calculations) and an appropriate
energy grid when calculating the LDOS. See the
:doc:`MALA publications <../about/publications>` for
examples of such values for other materials. Lastly, when calculating
the LDOS with ``pp.x``, make sure to set ``use_gauss_ldos=.true.`` in the
``inputpp`` section.


Data Conversion
###############

MALA can be used to process raw data into ready-to-use data fro the surrogate models
For this, the ``DataConverter`` class can be used; see example ``ex02_preprocess_data``.

Using input and output data
###########################

MALA divides provided data into two categories: ``Descriptor`` objects
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
E.g. in a script like this:

      .. code-block:: python

            import mala

            parameters = mala.Parameters()
            parameters.targets.descriptor_type = 'LDOS'

            # Creates a LDOS object via interface
            ldos = mala.Target(parameters)

            # Creates a LDOS object directly
            ldos = mala.LDOS(parameters)

            # Use the LDOS object to calculate the band energy from LDOS data.
            ldos_data = ldos.read_from_cube(...)
            band_energy = ldos.get_band_energy(ldos_data)


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
