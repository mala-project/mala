#!/bin/bash                                                 
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -p blake                                                         
#SBATCH -A athomps
#SBATCH --time=48:00:00
#SBATCH --job-name=fp_gen


# Run `python3 ldos_example.py --help` for more option information

NODES=4
RPN=1 
RANKS=$((${NODES}*${RPN}));

echo "Total ranks: ${RANKS}"

# 64 million grid pts
NXYZ="400"
#NXYZ="20"
NSTRING="n2048"
NSTRINGARG="--nstring ${NSTRING}"
#NSTRING=""
#NSTRINGARG=""

#EXE="mpirun -np ${RANKS} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
EXE="mpirun -np ${RANKS}" 


#TEMPS="300K 10000K 20000K 30000K"
#TEMPS="300K" 

#GCCS="0.1 0.2 0.4 0.6 0.8 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0"
#GCCS="0.4 0.6 0.8 1.0 2.0 3.0 4.0 6.0 7.0 8.0 9.0"
#GCCS="2.0"

echo $(date)
${EXE} python generate_fingerprints.py ${NSTRINGARG} --nxyz ${NXYZ} > logs/gen_fp${NSTRING}_n${NODES}_ranks${RANKS}_nxyz${NXYZ}.log
echo $(date) 

exit 0
