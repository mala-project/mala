
#### Sets up python env for the MLMM project

# Run on machines/testbeds with no root privs

PIP=pip3
PYTHON=python3

#${PIP} install --upgrade pip --user

#source load_mlmm_blake_modules.sh

#${PIP} install --user virtualenv

${PYTHON} -m venv ../mlmm_env

source ../mlmm_env/bin/activate

${PIP} install numpy 
${PIP} install scipy 
${PIP} install matplotlib
${PIP} install seaborn

# Use PyTorch
${PIP} install torch
${PIP} install torchvision

# SYNAPSE CUDA VERSION 
#${pip} install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Logging 
${PIP} install tensorboard

# Parallel support
${PIP} install horovod
${PIP} install mpi4py

# Atomic Simulation Environment
${PIP} install ase

# Numeric Tools
${PIP} install sklearn
${PIP} install mpmath




${PIP} list

