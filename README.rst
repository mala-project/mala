MALA
====

.. image:: https://github.com/mala-project/mala/actions/workflows/gh-pages.yml/badge.svg
    :target: https://mala-project.github.io/mala/

MALA tools (Materials Analysis and Learning) is a machine-learning
based framework to enable multiscale modeling by bypassing
computationally expensive density functional simulations. It is designed
as a python package. This repository is structured as follows:

.. code::

   ├── examples : contains useful examples to get you started with the package
   ├── install : contains scripts for setting up this package on your machine
   ├── mala : the source code itself
   ├── test : test scripts used during development, will hold tests for CI in the future
   └── docs : Sphinx documentation folder


Installation
------------

Please refer to :doc:`Installation of FESL <install/README>`.

Running
-------

You can familiarize yourself with the usage of this package by running
the examples in the ``example/`` folder.

References
----------

[1] Ellis, J. A., Cangi, A., Modine, N. A., Stephens, J. A., Thompson,
A. P., & Rajamanickam, S. (2020). Accelerating Finite-temperature
Kohn-Sham Density Functional Theory with Deep Neural Networks. arXiv
preprint `arXiv:2010.04905 <https://arxiv.org/abs/2010.04905>`_.
