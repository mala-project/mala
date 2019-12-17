#!/bin/bash                                                 
#SBATCH -N 16
#SBATCH -n 16
#SBATCH -p blake                                                         
#SBATCH -A johelli                                                   
#SBATCH --time=48:00:00                                        
#SBATCH --job-name=ldos_example


# Run `python3 ldos_example.py --help` for more option information

NODES=16
RANKS=${NODES}

echo "Total ranks: ${RANKS}"


# Model 1: Density estimation with activations
# Model 2: Density estimation without activations
# Model 3: LDOS estimation without LSTM
# Model 4: LDOS estimation with LSTM
# Model 5: LDOS estimation with bidirectional LSTM

#MODEL="1 2 3 4 5"
MODEL="5"


# 8 million grid pts
NXYZ="200"

BATCHSIZE="64" 
EPOCHS="20"

#EXE="mpirun -np ${RANKS} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
EXE="mpirun -np ${RANKS} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^tcp"



echo $(date)

for MDL in $MODEL
do
    for BS in $BATCHSIZE
    do
        echo "Running Model ${MDL}, NXYZ ${NXYZ}, BatchSize ${BS}, Epochs ${EPOCHS}"

        ${EXE} python3 ldos_example.py --model ${MDL} --nxyz ${NXYZ} --batch-size ${BS} --epochs ${EPOCHS} > logs/ldos_example_n${NODES}_rpn${RPN}_model${MDL}_nxyz${NXYZ}_batchsize${BS}_epochs${EPOCHS}.log
    done
done

echo $(date) 

exit 0
