# Installation of MALA

This document details the manual installation of the MALA package.


## Supported OS and python version

MALA itself is not bound to a certain python version, as long as the package requirements for MALA can be met by the
version of your choice. MALA has been successfully tested for python 3.8.x.
MALA can be run on Linux, macOS and Windows, so as long as you have a working
python installation with the required depdencies.
A list of machines on which MALA was tested can be found in [Successfully tested on](tested_systems).

## Using conda to manage dependencies (Recommended)

While installing MALA from `pip` will install required packages, we recommend
using [conda](https://docs.conda.io/en/latest/miniconda.html)  to install them 
beforehand. However, using `pip` to resolve dependencies works as well 
(see below). 

We provide four different environment files, depending on the architecture and
granularity of dependencies specified (find them in the `install` folder).
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

## Installation from pip

When installing MALA via `pip`, all necessary packages are downloaded, 
if they have not already been installed with e.g. `conda`. The exception to 
this is `torch`.  We don't want `torch` to be installed automatically, as 
users may have or want to work with special `torch` builds provided by their 
HPC infrastructure. If you install MALA manually from scratch, make sure there 
is a `torch` version available to python. For local testing, people
might want to install the CPU-only version using something like

```sh
$ pip install torch==1.7.1+cpu torchvision==0.8.2+cpu \
    torchaudio==0.7.2 -f \
    https://download.pytorch.org/whl/torch_stable.html
```

while a plain `pip install torch` will pull the much larger GPU version by
default. For other ways to install PyTorch you might also refer 
to <https://pytorch.org/>.

Afterwards, MALA can easily be installed via

```sh
$ cd ~/path/to/this/git/root/directory
$ pip install -e .[options]
```

The following options are available:
- `dev`: Installs `bump2version` which is needed to correctly increment 
  the version and thus needed for large code development
- `opt`: Installs `oapackage` and `pqkmeans`, so that the orthogonal array
  method may be used for hyperparameter optimization and clustered 
  training data sets are accesible, both of which may be relevant if you 
  plan to do large scale hyperparameter optimization
- `test`: Installs `pytest` which allows users to test the code
- `doc`: Installs all dependencies for building the documentary locally

Similar to `torch`, MALA also uses optional packages which we do not include 
in our general pip setup, as their configuration and/or installation is 
generally more specific to the machine you operate on. Namely, MALA can be 
used with:

* `lammps`: Enables the calculation of descriptors, see [the instructions on external modules](external_modules.rst).
* `total_energy`: Enables the calculation of the total energy, see [the instructions on external modules](external_modules.rst).
* `mpi4py`: Enables inference parallelization (not installed alongside other
            packages as installation may be very specific to your setup)
* `horovod`: Enables training parallelization (not installed alongside other
            packages as installation may be very specific to your setup)
* `minterpy`: Enables the calculation of minterpy descriptor using the [minterpy](https://github.com/casus/minterpy) code.
  
MALA can be used without these packages, an error will only occur when attempting
perform an operation these packages are crucial for. With the exception
of `lammps` and `total_energy`, these packages can be installed using
`pip`.

## Downloading and adding example data (Recommended)

The examples and tests need additional data to run. The MALA team provides a
data repository, that can be obtained from
<https://github.com/mala-project/test-data>. Please be sure to check out the
correct tag for the data repository, since the data repository itself is
subject to ongoing development as well.

Also make sure to have the [Git LFS](https://git-lfs.com/) installed on your
machine, since the data repository operates using Git LFS to handle large
binary files for example training data. 

Download data repository and check out correct tag:

```sh
$ git clone https://github.com/mala-project/test-data ~/path/to/data/repo
$ cd ~/path/to/data/repo
$ git checkout v1.6.0
```

Export the path to that repo by

```sh
$ export MALA_DATA_REPO=~/path/to/data/repo
```

This will be used by tests and examples.

## Build documentation locally (Optional)

Install the prerequisites (if you haven't already during the MALA setup).
```sh
$ pip install -r docs/requirements.txt
```

Afterwards, the documentation can be built via:

1. Change into `docs/` folder.
2. Run `make apidocs` on Linux/macOS or `.\make.bat apidocs` on Windows.
3. Run `make html` on Linux/macOS or `.\make.bat html` on Windows. This creates a `_build` folder inside `docs`. You may also want to use `make html SPHINXOPTS="-W"` sometimes. This treats warnings as errors and stops the output at first occurence of an error (useful for debugging rST syntax).
4. Open `docs/_build/html/index.html`.
5. Run `make clean` on Linux/macOS or `.\make.bat clean` on Windows. if required (e.g. after fixing erros) and building again
