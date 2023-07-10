.. _openpmd:

Storing data with OpenPMD
=========================

The current MALA default to save volumetric data are ``numpy`` objects/files.
However, ``numpy`` files do not store metadata, which is crucial when
attempting to build large-scale data-driven workflows. Furthermore, they
do not offer extensive compression capabilities.

To this an end, MALA now supports the `openPMD <https://github.com/openPMD>`_
standard. OpenPMD is a powerful standard/library that allows for the efficient
storage of volumetric data alongside relevant metadata, and further offers
capabilities for parallelization.

Currently, openPMD is tested by the MALA team in production and therefore
not the default option for data handling. Yet, MALA is fully compatible with
openPMD, and its use is highly encouraged. To do so, just replace the
``.npy`` file ending with a openPMD compliant file ending (e.g. ``.h5``) in
all instances of the ``DataConverter``, ``DataHandler`` and ``DataShuffler``
class, and specify ``openpmd`` where necessary; the workflows itself can
be left untouched. Specifically, set

      .. code-block:: python

            parameters = mala.Parameters()

            # Changes for DataConverter
            data_converter = mala.DataConverter(parameters)
            data_converter.convert_snapshots(...,
                                             naming_scheme="Be_snapshot*.h5")
            ...
            # Changes for DataHandler
            data_handler = mala.DataHandler(parameters)
            data_handler.add_snapshot("Be_snapshot0.in.h5", data_path,
                                       "Be_snapshot0.out.h5", data_path, "tr",
                                       snapshot_type="openpmd")
            ...
            # Changes for DataShuffler
            data_shuffler = mala.DataShuffler(parameters)
            # Data can be shuffle FROM and TO openPMD - but also from
            # numpy to openPMD.
            data_shuffler.add_snapshot("Be_snapshot0.in.h5", data_path,
                                        "Be_snapshot0.out.h5", data_path,
                                        snapshot_type="openpmd")
            data_shuffler.shuffle_snapshots(...,
                                            save_name="Be_shuffled*.h5")
