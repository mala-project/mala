# Installation of FESL

This document details the manual installation of the FESL package.

## Prerequisites

### Python packages

In order to run FESL you have to have the following packages installed:

* torch (PyTorch)
* numpy
* scipy
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

or just install the package along with dependencies

```sh
$ pip install -e .
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

Follow these steps to set up the necessary environment.
1. Create an environment `fesl`:
   ```
   conda env create -f environment.yaml
   ```
2. Activate the new environment:
   ```
   conda activate fesl
   ```
3. You can deactivate the environment with:
    ```
    conda deactivate
    ```
4. You can update your conda environment by making changes to environment.yaml:
    ```
    conda env update --name fesl  --file environment.yaml
    ```
###  LAMMPS (optional)

This workflow uses the Large-scale Atomic/Molecular Massively Parallel
Simulator (LAMMPS, https://lammps.sandia.gov/) for input data preprocessing. It
is not necessary to install LAMMPS if you are you using this workflow with
preprocessed data. If you need/want to install LAMMPS, please refer to
`INSTALL_LAMMPS.md`.

## Installing the FESL package

1. Make sure that all prerequisites are fulfilled (see above)
2. Install the package using

    ```
    cd ~/path/to/this/git/root/directory
    pip install -e .
    ```
    (note: the `-e` is absolutely crucial, so that changes in the code will be
    reflected system wide)
3. Go to the `examples` folder and run `ex0_verify_installation.py` to make
sure the setup was successful.
4. Enjoy!
