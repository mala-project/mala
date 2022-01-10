
# Set Blake modules
source load_mlmm_blake_modules.sh

# Set python3 environment with needed packages
source ../mlmm_env/bin/activate

# provide lammps shared lib for fp generation
export LD_LIBRARY_PATH=/ascldap/users/johelli/Code/mlmm/mlmm-ldrd-data/networks/build_lammps:$LD_LIBRARY_PATH



