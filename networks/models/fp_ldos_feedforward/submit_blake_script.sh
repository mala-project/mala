#!/bin/bash                                                 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p blake                                                         
#SBATCH -A johelli                                                   
#SBATCH --time=48:00:00                                        
#SBATCH --job-name=fp_ldos


# Run `python3 ldos_example.py --help` for more option information

NODES=1
RANKS=${NODES}

echo "Total ranks: ${RANKS}"


# Model 1: Density estimation with activations
# Model 2: Density estimation without activations
# Model 3: LDOS estimation without LSTM
# Model 4: LDOS estimation with LSTM
# Model 5: LDOS estimation with bidirectional LSTM

#MODEL="1 2 3 4 5"
MODEL="3"


# 8 million grid pts
NXYZ="200"

BATCHSIZE="64" 
EPOCHS="1"

#EXE="mpirun -np ${RANKS} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
EXE="mpirun -np ${RANKS} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^tcp"

OPTIONS="--gcc 3.0 --no-coords"

echo $(date)

for MDL in $MODEL
do
    for BS in $BATCHSIZE
    do
        echo "Running Model ${MDL}, NXYZ ${NXYZ}, BatchSize ${BS}, Epochs ${EPOCHS}, with options ${OPTIONS}"

        ${EXE} python3 ldos_example.py --model ${MDL} --nxyz ${NXYZ} --batch-size ${BS} --epochs ${EPOCHS} ${OPTIONS} > logs/ldos_example_n${NODES}_rpn${RPN}_model${MDL}_nxyz${NXYZ}_batchsize${BS}_epochs${EPOCHS}.log
    done
done

echo $(date) 

exit 0
