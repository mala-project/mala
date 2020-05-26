
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




# Use TF
#${PIP} install keras=2.2.5
#${PIP} install tensorflow==1.14.0
#${PIP} install tensorflow_datasets 
#${PIP} install tensorboard 

# Not Tested
#${PIP} install tensorflow-gpu
#${PIP} install tensorflow-gpu==2.0.0-rc0

${PIP} list

