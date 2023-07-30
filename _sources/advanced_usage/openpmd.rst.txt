.. _openpmd data:

Storing data with OpenPMD
=========================

The current MALA default to save volumetric data are ``numpy`` objects/files.
However, ``numpy`` files do not store metadata, which is crucial when
attempting to build large-scale data-driven workflows. Furthermore, they are
inherently non-parallel and do not offer extensive compression capabilities.

To this end, MALA now supports the `openPMD <https://github.com/openPMD>`_
standard. OpenPMD is a powerful standard/library that allows for the efficient
storage of volumetric data alongside relevant metadata, and further offers
capabilities for parallelization and a declarative runtime configuration
for compression options.

Currently, openPMD is tested by the MALA team in production and therefore
not the default option for data handling. Yet, MALA is fully compatible with
openPMD, and its use is highly encouraged. To do so, just replace the
``.npy`` file ending with a openPMD compliant file ending (e.g. ``.h5``) in
all instances of the ``DataConverter``, ``DataHandler`` and ``DataShuffler``
class, and specify ``openpmd`` where necessary; the workflows themselves can
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

For further information on the interaction with openPMD data,
please consult the `official documentation <https://openpmd-api.readthedocs.io>`_.
As a user of MALA, you will be mainly interested in the scientific tooling
that can read openPMD, e.g.:

* visualization and analysis, including an exploratory Jupyter notebook GUI: `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`_
* `yt-project <http://yt-project.org/doc/examining/loading_data.html#openpmd-data>`_ (`tutorial <https://gist.github.com/C0nsultant/5808d5f61b271b8f969d5c09f5ca91dc>`_)
* `ParaView <https://www.paraview.org/>`_ has a Python-based openPMD plugin that can be activated by opening a helper text file ending on ``.pmd`` that contains one line with the openPMD-api Series filename, e.g. ``data_%T.bp``
* `VisIt <https://github.com/openPMD/openPMD-visit-plugin>`_
* converter tools: `openPMD-converter <https://github.com/openPMD/openPMD-converter>`_
* full list of `projects using openPMD <https://github.com/openPMD/openPMD-projects>`_

If you intend to write your own post-processing routines, make sure to check out our `example files <https://github.com/openPMD/openPMD-example-datasets>`_ and the `formal, open standard <https://github.com/openPMD/openPMD-standard>`_ on openPMD.
