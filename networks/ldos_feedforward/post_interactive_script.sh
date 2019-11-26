
# Ensure that you have salloc'd enough nodes and that modules/python env is correct

# Run `python3 ldos_example.py --help` for more option information

NODES="1 2 4"
RANKS=${NODES}

echo "Total ranks: ${RANKS}"


# Model 1: Density estimation with activations
# Model 2: Density estimation without activations
# Model 3: LDOS estimation without LSTM
# Model 4: LDOS estimation with LSTM
# Model 5: LDOS estimation with bidirectional LSTM

#MODEL="1 2 3 4 5"
MODEL="4 5"
#MODEL="1"


# 8 million grid pts
NXYZ="20"
#NXYZ="200"

BATCHSIZE="64" 
EPOCHS="1"


echo $(date)

for RNK in $RANKS
do
    EXE="mpirun -np ${RNK} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
    for MDL in $MODEL
    do
        for BS in $BATCHSIZE
        do
            echo "Running Model ${MDL}, Nodes ${RNK}, NXYZ ${NXYZ}, BatchSize ${BS}, Epochs ${EPOCHS}"

            ${EXE} python3 ldos_example.py --model ${MDL} --nxyz ${NXYZ} --batch-size ${BS} --epochs ${EPOCHS} > "./logs/ldos_example_model${MDL}_nodes${RNK}_nxyz${NXYZ}_batchsize${BS}_epochs${EPOCHS}.txt"
        done
    done
done

echo $(date) 

exit 0
