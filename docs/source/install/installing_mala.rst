Installing MALA
===============

Prerequisites
**************

MALA supports any Python version starting from ``3.10.4``. No upper limit on
Python versions are enforced. The most recent *tested* version is ``3.10.12``.

Installing the Python library
*****************************

MALA requires ``torch`` (https://pytorch.org). We install the latest
GPU-enabled version (see ``requirements.txt``),
unless you have ``torch`` already installed (for example a version that supports
AMD's ROCm a specific CUDA version).

To install the MALA package

* Download the MALA repository, e.g., with ``git clone git@github.com:mala-project/mala.git``
* Change into the directory you cloned the repository to
* Install MALA via ``pip install -e .[options]``

The following options are available:

- ``dev``: Installs ``bump2version`` which is needed to correctly increment
  the version and thus needed for large code development (developers)
- ``opt``: Installs ``oapackage``, so that the orthogonal array
  method may be used for large scale hyperparameter optimization (advanced users)
- ``mpi``: Installs ``mpi4py`` for MPI parallelization (advanced users)
- ``test``: Installs ``pytest`` which allows users to test the code (developers)
- ``doc``: Installs all dependencies for building the documentary locally (developers)

Downloading and adding example data (Recommended)
*************************************************

The examples and tests need additional data to run. The MALA team provides a
`data repository <https://github.com/mala-project/test-data>`_. Please be sure
to check out the correct tag for the data repository, since the data repository
itself is subject to ongoing development as well.

* Download data repository and check out correct tag:

.. code-block:: bash

    git clone https://github.com/mala-project/test-data ~/path/to/data/repo
    cd ~/path/to/data/repo
    git checkout 

* Export the path to that repo by ``export MALA_DATA_REPO=~/path/to/data/repo``

This will be used by tests and examples.

Build documentation locally (Optional)
**************************************

* Install the prerequisites (if you haven't already during the MALA setup) via ``pip install -r docs/requirements.txt``
* Change into ``docs/`` folder.
* Run ``make apidocs`` on Linux/macOS or ``.\make.bat apidocs`` on Windows.
* Run ``make html`` on Linux/macOS or ``.\make.bat html`` on Windows. This creates a ``_build`` folder inside ``docs``. You may also want to use ``make html SPHINXOPTS="-W"`` sometimes. This treats warnings as errors and stops the output at first occurence of an error (useful for debugging rST syntax).
* Open ``docs/_build/html/index.html``.
* Run ``make clean`` on Linux/macOS or ``.\make.bat clean`` on Windows. if required (e.g. after fixing erros) and building again
