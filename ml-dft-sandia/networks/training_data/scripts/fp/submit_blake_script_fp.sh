#!/bin/bash                                                 
#SBATCH -N 4   
#SBATCH -n 32
#SBATCH -p blake                                                         
#SBATCH -A johelli                                                   
#SBATCH --time=24:00:00
#SBATCH --job-name=fp_gen


# Run `python3 ldos_example.py --help` for more option information

NODES=4
RPN=8
RANKS=$((${NODES}*${RPN}));

echo "Total ranks: ${RANKS}"

# 8 million grid pts
NXYZ="200"
#NXYZ="20"

#EXE="mpirun -np ${RANKS} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
EXE="mpirun -np ${RANKS}" 


#TEMPS="300K 10000K 20000K 30000K"
TEMPS="300K" 

#GCCS="0.1 0.2 0.4 0.6 0.8 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0"
GCCS="0.4 0.6 0.8 1.0 2.0 3.0 4.0 6.0 7.0 8.0 9.0"
#GCCS="2.0"

echo $(date)

for T in $TEMPS
do
    for G in $GCCS
    do
        echo "Running fingerprint generation for temp ${T} and gcc ${G} with nxyz ${NXYZ}" 

        ${EXE} python3 generate_fingerprints.py --temp ${T} --gcc ${G} --nxyz ${NXYZ} > logs/gen_fp_n${NODES}_ranks${RANKS}_temp${T}_gcc${G}_nxyz${NXYZ}.log
    done
done

echo $(date) 

exit 0
