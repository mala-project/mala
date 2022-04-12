Postprocessing the LDOS
=======================

MALA provides routines to calculate quantities of interests from the physical
data such as the LDOS, DOS and electronic density. The latter are not directly
predicted by MALA. In practice, they are either obtained via the LDOS, or
come directly from a DFT calculation. Basic postprocessing can be done via

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

But a wider range of properties is available, see e.g. ``ex03_postprocess_data``.
For a full list, consult the API reference.
