# Installation of MALA

This document details the manual installation of the MALA package.

## Prerequisites

### Python version

MALA itself is not bound to a certain python version, as long as the package requirements for MALA can be met by the
version of your choice. MALA has been successfully tested for python 3.8.5.
If you intend to use the optional Quantum Espresso total energy module or LAMMPS and install these software packages
yourself the necessary python bindings will be created for your python version. If you operate on a machine that
provides pre-installed versions of these packages (e.g. on an HPC cluster), you need to adhere to the python version
for which these python bindings have been created. A list of machines on which MALA was tested can be found in
[Successfully tested on](tested_systems).

### Python packages

In order to run MALA you have to have the following packages installed:

* torch (PyTorch)
* numpy
* scipy
* oapackage
* tensorboard
* optuna
* ase
* mpmath

See also the `requirements.txt` file.
You can install each Python package with

```sh
$ pip install packagename
```

or all with

```sh
$ pip install -r requirements.txt
```

or just install the package along with its basic dependencies

```sh
$ pip install -e .
```

In order to install additional dependencies (enabling optional features, building documentaion locally, testing etc.) do

```sh
$ pip install -e .[opt,test,doc]
```


See below for what the `-e` option does.

Note that we exclude `torch` in `requirements.txt` We don't want torch to be
installed automatically, just note that we need it. For local testing, people
might want to install the CPU-only version using something like

```sh
$ pip install torch==1.7.1+cpu torchvision==0.8.2+cpu \
    torchaudio==0.7.2 -f \
    https://download.pytorch.org/whl/torch_stable.html
```
while a plain `pip install torch` will pull the much larger GPU version by
default.

For other ways to install PyTorch you might also refer to <https://pytorch.org/>.

#### Conda environment (optional)

Alternatively to pip, you can install the required packages within a
[conda](https://docs.conda.io/en/latest/miniconda.html) environment.
We provide four different environment files, depending on the architecture and
ganularity of dependencies specified (find them in the `install` folder).
We use the explicit environment files for our reproducible and tested builds. Feel free to
start off with a basic environment file specifing only the direct dependencies (these
files will likely install newer versions of (sub-)dependencies and we cannot guarantee everything to work).

| dependency granularity: | basic                           | explicit                   |
|-------------------------|---------------------------------|----------------------------|
| CPU                     | `mala_cpu_base_environment.yml` | `mala_cpu_environment.yml` |
| GPU                     | `mala_gpu_base_environment.yml` | `mala_gpu_environment.yml` |

Follow the steps below to set up the necessary environment.

1. (optional) If needed, specify the python version you want to use in this
environment by replacing `3.6` with your desired version in `environment.yml`:
   ```yaml
   dependencies:
     - python>=yourversionhere
   ```
2. Create an environment `mala`:
   ```sh
   $ conda env create -f environment.yml
   ```
3. Activate the new environment:
   ```sh
   $ conda activate mala
   ```
4. You can deactivate the environment with:
    ```sh
    $ conda deactivate
    ```
5. You can update your conda environment by making changes to environment.yml:
    ```sh
    $ conda env update --name mala  --file environment.yml
    ```
###  LAMMPS (optional)

This workflow uses the Large-scale Atomic/Molecular Massively Parallel
Simulator [LAMMPS](https://lammps.sandia.gov/) for input data preprocessing. It
is not necessary to install LAMMPS if you are you using this workflow with
preprocessed data. If you need/want to install LAMMPS, please refer to
[Setting up LAMMPS](INSTALL_LAMMPS.md).

### Total energy module (optional)

It is possible to utilize Quantum Espresso to calculate the total energy of a given system using python bindings.
This provides an additional post-processing capability, but is not necessarily needed for using this workflow.
If you want to use this module, please refer to [Python bindings to Quantum Espresso](INSTALL_TE_QE.md).


## Installing the MALA package

1. Make sure that all prerequisites are fulfilled (see above)
2. Install the package using

    ```sh
    $ cd ~/path/to/this/git/root/directory
    $ pip install -e .
    ```
    (note: the `-e` is absolutely crucial, so that changes in the code will be
    reflected system wide)
3. (Optional): Download example data (see below) to run the examples.

### Build documentation locally (optional)

Install the prerequisites:
```sh
$ pip install -r docs/requirements.txt
```

1. Change into `docs/` folder.
2. Run `make apidocs`.
3. Run `make html`. This creates a `_build` folder inside `docs`. You may also want to use `make html SPHINXOPTS="-W"` sometimes. This treats warnings as errors and stops the output at first occurence of an error (useful for debugging rST syntax).
4. Open `docs/_build/html/index.html`.
5. `make clean` if required (e.g. after fixing erros) and building again


## Downloading and adding example data

The examples and tests need additional data to run. The MALA team provides a data repository, that can be downloaded
from <https://github.com/mala-project/test-data>. Please be sure to check out the correct tag for the data repository,
since the data repository itself is subject to ongoing development as well. After downloading the correct revision of
the data repository, it needs to be linked with MALA.

1. Download data repository and check out correct tag.

   ```sh
   $ git clone https://github.com/mala-project/test-data ~/path/to/data/repo
   $ cd ~/path/to/data/repo
   $ git checkout v0.4.0
   ```

2. Link MALA and data repository.

   ```sh
   $ cd ~/path/to/mala/root/directory
   $ bash install/data_repo_link/link_data_repo.sh ~/path/to/data/repo
   ```

   Afterwards, check that files named `data_repo_path.py` have been generated.

   ```sh
   $ find . -name data_repo_path.py
   ./install/data_repo_link/data_repo_path.py
   ./test/data_repo_path.py
   ./examples/data_repo_path.py
   ```
