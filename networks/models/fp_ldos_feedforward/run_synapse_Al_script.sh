
EXE="horovodrun --gloo-timeout-seconds 600"
#EXE="mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
#EXE=mpirun
#EXE=mpiexec

PYT=python3
DRVR=fp_ldos_driver.py


if [ $1 ]
then
    echo "$1 GPUS"
    GPUS=$1
else
    echo "Default 2 GPUs"
    GPUS=2
fi


### Input/Output

#DS=random
DS=fp_ldos

# Test Case
MAT=Al
TMP=298K
#TMP=933K
GC=2.699
NXYZ=200

FPL=94
LDOSL=250

#SSHOT=5
#SSHOT=3
#SSHOTS="10"
#SSHOTS="4 6 8 10"

SSHOT=4

#CLUSTERS="300"
CLUSTER=200

#CTRS=".01 .05 .1 .2 .4"
CTR=.05

#CSRS=".01 .05 .1 .2 .4"
CSR=.4


### Optimizer/Training

#EPCS=250
#EPCS=100
#EPCS=10
EPCS=2

#TBSS="4000 8000 16000 32000" 
TBS=4000
#BATS="16 32 64"
#BATS="4000"
BS=4000
#BS=4000

#LRS=".0005 .0001 .00005 .00001"
#LRS=".0001 .00005 .00001"
LR=.00001
ES=.99999
EP=8
OP=4

#LIL="6"
LIL=1

NUMW=1
#NUMW=4
#NUMWS="0 1 2 3 5 6 7 8"
NUMTWS="1 2 4 8 16"

# Network
WID=800
#WID=300
MIDLYS=2
#MIDLYS=3
AE=.8

#   --model-lstm-network \
#   --adam \
#   --skip-connection \

#   --big-charm-data \
#   --big-clustered-data \


for NUMTW in $NUMTWS
do

    ${EXE} -np ${GPUS} ${PYT} ${DRVR} \
        --dataset ${DS} \
        --epochs ${EPCS} \
        --big-charm-data \
        --num-clusters ${CLUSTER} \
        --cluster-train-ratio ${CTR} \
        --cluster-sample-ratio ${CSR} \
        --no-pinned-memory \
        --no-testing \
        --adam \
        --material ${MAT} \
        --nxyz ${NXYZ} \
        --temp ${TMP} \
        --gcc ${GC} \
        --batch-size ${BS} \
        --test-batch-size ${TBS} \
        --fp-length ${FPL} \
        --ldos-length ${LDOSL} \
        --optim-patience ${OP} \
        --early-patience ${EP} \
        --early-stopping ${ES} \
        --lr ${LR} \
        --lstm-in-length ${LIL} \
        --ff-mid-layers ${MIDLYS} \
        --ff-width ${WID} \
        --ae-factor ${AE} \
        --num-snapshots ${SSHOT} \
        --no-coords \
        --num-data-workers ${NUMW} \
        --num-test-workers ${NUMTW} \
        --calc-training-norm-only \
        --fp-row-scaling \
        --fp-standard-scaling \
        --ldos-norm-scaling \
        --ldos-max-only \
        2>&1 | tee ./logs/fp_ldos_synapse_${MAT}_${GPUS}gpus_${TMP}_${GC}gcc_${FPL}fp_${LDOSL}ldos_${NXYZ}nxyz_${RANDOM}${RANDOM}.log

done


#### Additional options

#        --save-training-data \

#        --fp-row-scaling \
#        --fp-standard-scaling \
#        --ldos-norm-scaling \
#        --ldos-max-only \

#        --power-spectrum-only \

#        --no-hidden-state \



#        --ldos-max-only \

#        --fp-row-scaling \
#        --ldos-row-scaling \
#        --fp-standard-scaling \
#        --ldos-standard-scaling \
#        --fp-log \
#        --ldos-log \









