#!/bin/bash -l
#SBATCH --job-name=ace_test
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH -A casus
#SBATCH -p rome

module load cmake/3.26.1
module load gcc/9.2.0
python grid_coeff.py
/home/callow46/lammps/build/lmp -in in.grid.ace > tim_ace_test.log
