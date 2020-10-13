
#### Sets up python env for the MLMM project

# Run on machines/testbeds with no root privs

PIP=pip3
PYTHON=python3

#${PIP} install --upgrade pip --user

#source load_mlmm_blake_modules.sh

#${PIP} install --user virtualenv

${PYTHON} -m venv ../mlmm_env

source ../mlmm_env/bin/activate

${PIP} install numpy==1.18.4
${PIP} install scipy==1.4.1
${PIP} install matplotlib==3.1.3
${PIP} install seaborn==0.10.0

# Use PyTorch
${PIP} install torch==1.5.0
${PIP} install torchvision==0.6.0

# SYNAPSE CUDA VERSION 
#${pip} install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Logging 
${PIP} install tensorboard==2.1.0

# Parallel support
${PIP} install horovod==0.19.2
${PIP} install mpi4py==3.0.3

# Atomic Simulation Environment
${PIP} install ase==3.19.0

# Numeric Tools
${PIP} install scikit-learn==0.19.2
${PIP} install mpmath==1.1.0
${PIP} install pqkmeans==1.0.4
${PIP} install sympy==1.6

# Outside Network Packages
#${PIP} install transformers==3.2.0
#${PIP} install e3nn

${PIP} list

