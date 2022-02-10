Data preprocessing
===================

Data Conversion
###############

Data generation for MALA is done by electronic structure calculations using
appropriate simulation software. The raw outputs of such calculations
are atomic positions and the LDOS, although often as separate cube files.
MALA can be used to process this raw data into ready-to-use data fro the surrogate models
For this, the `DataConverter` class can be used; see example `ex02_preprocess_data`.

Descriptors
***********

The input data for MALA are, in theory, the atomic positions. As MALA
oeprates "per-grid-point" manner, information from the atomic positions have
to be present on the entire grid of the simulation cell. This is done by
calculating descriptors on the grid. Currently, only SNAP descriptors are
supported. MALA uses LAMMPS to calculate these SNAP descriptors.

Targets
*******

MALA is optimized for the usage of the LDOS (local density of states) as
target quantity. The LDOS gives the DOS (density of states) at each grid point,
and thus gives information on the energy-grid as well as the 3D grid.
The LDOS can be used to :doc:`efficiently calculate quantities of interest.
<postprocessing>` MALA provides parsing routines to read the LDOS from
DFT calculations.

Data Scaling
############

An additional step of preprocessing is scaling the data before a model is
trained. This is done automatically in the ``DataHandler`` class, using the
methods requested via the ``mala.Parameters.data.input_rescaling_type`` and
``mala.Parameters.data.output_rescaling_type`` keywords. Currently supported here
are:

* "None": No normalization is applied.

* "standard": Standardization (Scale to mean 0, standard deviation 1)

* "normal": Min-Max scaling (Scale to be in range 0...1)

* "feature-wise-standard": Row Standardization (Scale to mean 0, standard deviation 1)

* "feature-wise-normal": Row Min-Max scaling (Scale to be in range 0...1)

Internally, the ``DataScaler`` class is used. The objects of this class
can be saved and loaded later for e.g. inference or to minimize calculation
time for multiple ML experiments using the same set of data.
Data scaling will always be done using the training data only.
