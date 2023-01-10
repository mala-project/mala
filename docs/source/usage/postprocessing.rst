Postprocessing the LDOS
=======================

MALA provides routines to calculate quantities of interests from the physical
data such as the LDOS, DOS and electronic density. The latter are not directly
predicted by MALA. In practice, they are either obtained via the LDOS, or
come directly from a DFT calculation. MALA implements two ways of accessing
observables and derived quantities from ab-initio data.

Cached access (recommended)
---------------------------

Mala `Target` objects provide several properties corresponding to physical
quantities. If a Target object is set up to contain electronic structure data,
then accessing properties will calculate them if needed, and provide cached
values if not. This method of access is preferred as it ensures consistency
and efficiency.
For example, one can do:

      .. code-block:: python

            import mala

            parameters = mala.Parameters()
            parameters.targets.descriptor_type = 'LDOS'

            # Creates a LDOS object directly.
            # This object will contain LDOS data.
            ldos = mala.LDOS.from_cube_file(parameters, path_to_cube_fiöle)

            # The first command will calculate the Fermi energy because it
            # has not been calculated yet. In the second command, the band
            # energy will be calculated, and the cached Fermi energy will
            # be used for that.
            fermi_energy = ldos.fermi_energy
            band_energy = ldos.band_energy

            # Uncaching the properties makes it possible to have them be
            # calculated anew (because e.g. temperatures have changed)
            ldos.uncache_properties()

A wider range of properties is available, see e.g. ``ex03_postprocess_data``.
For a full list, consult the API reference.

Direct access
-------------

For direct access to properties several `get_some_property()` functions exist.
They offer a more fine-granular access to several calculation parameters,
but put efficiency and consistency in the users hands. Usage is similar to
the cached access:

      .. code-block:: python

            import mala

            parameters = mala.Parameters()
            parameters.targets.descriptor_type = 'LDOS'

            # Creates a LDOS object directly, without data.
            ldos = mala.LDOS(parameters)

            # To perform calculations, one now has to read in data first.
            ldos.read_from_cube(...)

            # Use the LDOS object to calculate the band energy from LDOS data.
            # Several keyword arguments are possible here.
            band_energy = ldos.get_band_energy(ldos.ldos, ....)
