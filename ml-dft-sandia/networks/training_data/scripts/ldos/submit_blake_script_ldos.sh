#!/bin/bash                                                 
#SBATCH -N 1                                                  
#SBATCH -p blake                                                         
#SBATCH -A johelli                                                   
#SBATCH --time=12:00:00                                        
#SBATCH --job-name=ldos_parse


echo $(date)

python3 parse_ldos.py > logs/ldos_parse_300K.log

echo $(date) 

exit 0
