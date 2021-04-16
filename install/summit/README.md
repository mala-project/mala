# Installing MALA on Summit

## Prerequisites 

### SWIG

The default Summit's SWIG is 2.0.10, which is insufficient for MALA. To install current SWIG, follow the instructions on http://www.swig.org/svn.html (don't forget to specify actual path to ):
1. `$ git clone https://github.com/swig/swig.git`
2. `$ cd swig`
3. `$ ./autogen.sh`
4. `$ ./configure --prefix=/.../swig` (replace `...` with actual absolute path to `swig` directory)
5. `$ make`
6. `$ make install`

To direct Summit to the installed SWIG, add it to PATH:

4. `$ PATH = /.../swig/bin` (replace `...` with actual absolute path to `swig` directory)

### Python packages

It is highly recommended to install MALA prerequisites, including pytorch, using conda environment from one of the prepared files `mala_summit_cpu_base_environment.yml`, `mala_summit_cpu_environment.yml`, `mala_summit_gpu_base_environment.yml`, `mala_summit_gpu_environment.yml`. For example, to install a CPU-only version at the basic granularity level, you would type:

5. `$ conda env create -f mala_summit_cpu_base_environment.yml`
6. `$ source activate mala-cpu` (environment name will be shown by conda at the end of previous step, also can be found on line 1 of `mala_..._environment.yml`)

## Installation

7. `$ git clone`
8. `$ cd mala`
9. `$ pip install -e . --user`

## Testing the installation

A simplistic check of whether installation was a success is to ensure that your Python environment can execute `import mala`. To actually test MALA functionality, you can:

10. Download example data from https://gitlab.hzdr.de/multiscale-wdm/surrogate-models/fesl/data
11. `$ bash install/data_repo_link/link_data_repo.sh` (link the data to MALA installation)
12. `$ python examples/ex00_verify_installation.py`
