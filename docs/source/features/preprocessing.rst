Preprocessing
==============

Descriptors
***********

The input data for FESL are, in theory, the atomic positions. As FESL
oeprates "per-grid-point" manner, information from the atomic positions have
to be present on the entire grid of the simulation cell. This is done by
calculating descriptors on the grid. Currently, only SNAP descriptors are
supported. FESL uses LAMMPS to calculate these SNAP descriptors.

Targets
***********

FESL is optimized for the usage of the LDOS (local density of states) as
target quantity. The LDOS gives the DOS (density of states) at each grid point,
and thus gives information on the energy-grid as well as the 3D grid.
The LDOS can be used to :doc:`efficiently calculate quantities of interest.
<postprocessing>` FESL provides parsing routines to read the LDOS from
DFT calculations.
