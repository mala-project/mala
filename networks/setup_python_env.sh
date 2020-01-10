
#### Sets up python env for the MLMM project

# Run on machines/testbeds with no root privs

#pip3 install --upgrade pip --user

source load_mlmm_blake_modules.sh

pip3 install --user virtualenv

python3 -m venv mlmm_env

source mlmm_env/bin/activate

pip3 install numpy 
pip3 install scipy 
pip3 install matplotlib
pip3 install seaborn
pip3 install tensorboard

# Use PyTorch
pip3 install torch
pip3 install torchvision

# Parallel support
pip3 install horovod
pip3 install mpi4py

# Atomic Simulation Environment
pip3 install ase



# Use TF
#pip3 install keras=2.2.5
#pip3 install tensorflow==1.14.0
#pip3 install tensorflow_datasets 
#pip3 install tensorboard 

# Not Tested
#pip3 install tensorflow-gpu
#pip3 install tensorflow-gpu==2.0.0-rc0

pip3 list

