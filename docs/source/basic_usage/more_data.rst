Data generation and conversion
===============================

MALA operates on volumetric data. Volumetric data is stored in binary files.
By default - and discussed here, in the introductory guide - this
means ``numpy`` files (``.npy`` files). Advanced data storing techniques
are :ref:`also available <openpmd data>`.

Data generation
###############

Data generation for MALA is done by electronic structure calculations using
appropriate simulation software. The raw outputs of such calculations
are atomic positions and the LDOS, the latter usually as multiple individual
.cube files.

Currently, only Quantum ESPRESSO has been tested for preprocessing.
Starting with version 7.2, any version of Quantum ESPRESSO can be used to
create data for MALA. In order to do so

1. Perform a regular DFT calculation using ``pw.x``
2. Calculate the LDOS using ``pp.x``

Make sure to use enough k-points in the DFT calculation (LDOS sampling
requires denser k-grids then regular DFT calculations) and an appropriate
energy grid when calculating the LDOS. See the `initial MALA publication <https://www.doi.org/10.1103/PhysRevB.104.035120>`_
for more information on this topic. Lastly, when calculating
the LDOS with ``pp.x``, make sure to set ``use_gauss_ldos=.true.`` in the
``inputpp`` section.


Data conversion
###############

Once you have performed the necessary simulations, you will have to calculate
the volumetric descriptor field from the atomic positions and transform
the LDOS into a format useable by MALA.

MALA can be used to process raw data into ready-to-use data for ML-DFT model
creation. For this, the ``DataConverter`` class can be used, as also shown
in the example ``basic/ex03_preprocess_data``.
The first thing when converting data is to select how the data should be
processed. Up until now, MALA operates with bispectrum descriptors as
input data (=descriptors) and LDOS as output data (=targets). Their
calculation is calculated via

      .. code-block:: python

            parameters = mala.Parameters()
            # Bispectrum parameters.
            parameters.descriptors.descriptor_type = "Bispectrum"
            parameters.descriptors.bispectrum_twojmax = 10
            parameters.descriptors.bispectrum_cutoff = 4.67637
            # LDOS parameters.
            parameters.targets.target_type = "LDOS"
            parameters.targets.ldos_gridsize = 11
            parameters.targets.ldos_gridspacing_ev = 2.5
            parameters.targets.ldos_gridoffset_ev = -5

For the LDOS, these parameters are determined by the electronic structure
simulation. Namely, ``ldos_gridsize`` governs how many discretized energy
values are included in the energy grid upon which the LDOS is sampled,
``ldos_gridspacing_ev`` governs how far these values are apart and
``ldos_gridoffset_ev`` determines the lowest energy value sampled. These values
are chosen for the ``pp.x`` simulation and have to be given here.

For the bispectrum calculation, ``bispectrum_cutoff`` gives the radius of
the cutoff sphere from which information on the atomic structure is incoporated
into the bispectrum descriptor vector at each point in space, whereas
``bispectrum_twojmax`` governs the dimensionality of the bispectrum
representation at each point in space. If it is set to, e.g., 10, 91 components
are used at each point in space to encode the atomic structure.

The values for the bispectrum descriptors have to be chosen such
that the corresponding descriptors accurately represent atomic enviroment.
:ref:`In the advanced example section <tuning descriptors>`, it is shown
how these values can be determined by a newly developed method called
`ACSD <https://doi.org/10.1088/2632-2153/ac9956>`_.

After selecting these options, we have to create a ``DataConverter`` object
and fill it with data, e.g., by

      .. code-block:: python

            data_converter = mala.DataConverter(parameters)
            outfile = os.path.join(data_path, "Be_snapshot0.out")
            ldosfile = os.path.join(data_path, "cubes/tmp.pp*Be_ldos.cube")

            data_converter.add_snapshot(descriptor_input_type="espresso-out",
                                        descriptor_input_path=outfile,
                                        target_input_type=".cube",
                                        target_input_path=ldosfile,
                                        additional_info_input_type="espresso-out",
                                        additional_info_input_path=outfile,
                                        target_units="1/(Ry*Bohr^3)")

The ``add_snapshot`` function can be called multiple times to add
multiple snapshots to MALA.
For regular Quantum ESPRESSO calculations, the ``descriptor_input_type``
and ``target_input_type`` will always be ``"espresso-out"`` and ``".cube"``,
respectively, and the ``target_units`` will always be ``"1/(Ry*Bohr^3)"``.
The paths have to be modified accordingly. ``additional_info_input_*`` refers
to the calculation output file - MALA provides an interface to condense
the entire, verbose simulation output to ``.json`` files for further
processing. In the preceding section, we had to specify calculation output
files a number of times - instead, we can use the reduced ``.json`` files
if we let them be created by the ``DataConverter`` class.

Once data is provided, the conversion itself is simple.

      .. code-block:: python

            data_converter.convert_snapshots(descriptor_save_path="./",
                                             target_save_path="./",
                                             additional_info_save_path="./",
                                             naming_scheme="Be_snapshot*.npy",
                                             descriptor_calculation_kwargs=
                                             {"working_directory": data_path})
            # You can also provide only one path
            # data_converter.convert_snapshots(complete_save_path="./",
            #                                  naming_scheme="Be_snapshot*.npy",
            #                                  descriptor_calculation_kwargs=
            #                                  {"working_directory": data_path})

The ``convert_snapshots`` function will convert ALL snapshots added via
``add_snapshot`` and save the resulting volumetric numpy files to the
provided paths. You can either provide separate paths for the separate types
of data or give one complete path, ``complete_save_path``, depending on your
personal preference. Fine-granular access
to the calculators is enabled via the ``descriptor_calculation_kwargs`` and
``target_calculation_kwargs`` arguments, but usually not needed.
