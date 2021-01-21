module rm gcc
module rm python
module rm openmpi

module load python/3.7.3
module load openmpi/2.1.2/gcc/7.2.0
module load cmake/3.12.3

module list

#### Sets up python env for the MLMM project

# Run on machines/testbeds with no root privs

PIP=pip3
PYTHON=python3
PIPFLAGS='--no-cache-dir'

export HOROVOD_WITH_GLOO=1

#${PIP} install ${PIPFLAGS} --upgrade pip --user

#source load_mlmm_blake_modules.sh

#${PIP} install ${PIPFLAGS} --user virtualenv

${PYTHON} -m venv ../mlmm_env

source ../mlmm_env/bin/activate

${PIP} install ${PIPFLAGS} numpy==1.18.4
${PIP} install ${PIPFLAGS} scipy==1.4.1
${PIP} install ${PIPFLAGS} matplotlib==3.1.3
${PIP} install ${PIPFLAGS} seaborn==0.10.0

# Use PyTorch
${PIP} install ${PIPFLAGS} torch==1.5.0
${PIP} install ${PIPFLAGS} torchvision==0.6.0

# SYNAPSE CUDA VERSION 
#${pip} install ${PIPFLAGS} torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Logging 
${PIP} install ${PIPFLAGS} tensorboard==2.1.0

# Parallel support
${PIP} install ${PIPFLAGS} horovod==0.19.2
${PIP} install ${PIPFLAGS} mpi4py==3.0.3

# Atomic Simulation Environment
${PIP} install ${PIPFLAGS} ase==3.19.0

# Numeric Tools
${PIP} install ${PIPFLAGS} scikit-learn==0.19.2
${PIP} install ${PIPFLAGS} mpmath==1.1.0
${PIP} install ${PIPFLAGS} pqkmeans==1.0.4
${PIP} install ${PIPFLAGS} sympy==1.6

# Outside Network Packages
#${PIP} install ${PIPFLAGS} transformers==3.2.0
#${PIP} install ${PIPFLAGS} e3nn

${PIP} list

source ../mlmm_env/bin/activate
