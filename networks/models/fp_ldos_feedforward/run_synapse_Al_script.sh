
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

if [ $5 ]
then
	LDOSL=$5
else
	LDOSL=250
fi
### Input/Output

#DS=random
DS=fp_ldos




###############################################################################
###### PAPER -- Al 298K Optimized Case
MAT=Al
TMP=298K
GC=2.699
NXYZ=200
FPL=94
#LDOSL=250 #250

# Train - 1 snapshot, Valid - 1, Test - 1 
SSHOT=3
# Snapshot offset, train = 0th, validation = 1st, test = 2nd
SNPOFF=0

# Train Batch Size
BS=1000
# Valid/Test Batch Size
TBS=4000

# Learning Rate
#LR=.00001
LRS=".00001"
## Network Parameters

# Layer Width
WID=800

# Number of Mid Layers
MIDLYS=2

# Last Layer Scaling (e.g if LIL=2 and output_layer=250, then previous_layer=500)
LIL=1


###############################################################################
###### PAPER -- Al 933K Optimized Case (Hybrid/Liquid/Solid)
#MAT=Al
#TMP=933K
#GC=2.699
#NXYZ=200
#FPL=94
#LDOSL=250

## Train - 8 snapshots, Valid - 1, Test - 1 
#SSHOT=10

### Snapshot offset

## Hybrid Model (6,7,8,9 Liquid / 10,11,12,13 Solid) Training, (14 Solid) Validation, (15 Solid) Test
#SNPOFF=6

## Liquid Model (0-7 Liquid) Training, (8) Validation, (9) Test
#SNPOFF=0

## Solid Model (10-17 Solid) Training, (18) Validation, (19) Test
#SNPOFF=10


## Train Batch Size
#BS=1000
## Valid/Test Batch Size
#TBS=4000

## Learning Rate
#LR=.00005

### Network Parameters

## Layer Width
#WID=4000

## Number of Mid Layers
#MIDLYS=2

## Last Layer Scaling (e.g if LIL=2 and output_layer=250, then previous_layer=500)
#LIL=1







###############################################################################
###### General Parameters


## Optimizer/Training

# Max Epochs
EPCS=5000

# Early Stopping Patience
EP=5 #8
# Early Stopping Sufficient Decrease
ES=.99999

# Optimizer Learning Rate Schedule Patience
OP=4

## Accelerating / Experimental (Add "--big-clustered-data" option)

# Number of PQ k-means Clusters 
CLUSTER=400
# PQ k-means Sample percentage (Higher is better k-means approximation)
CSR=.1

# Cluster/Train ratio (Higher is more samples trained per epoch)
CTR=.25


## Number of Data-Workers, if using big-charm-data use NUMW=1
# Train
#NUMW=1
NUMW=16

# Valid/Test
NUMTW=16
 
if [ $2 ]
then
	FTST=$2
else
	FTST=0
fi

if [ $3 ]
then
	FTSP=$3
else
	FTSP=250
fi

if [ $4 ]
then
	ID=$4
else
	ID=${RANDOM}
fi

#FTST=0 #50
#FTSP=150 #150






## Big Data Case (i.e. many training snapshots)
#   --big-charm-data \

## Big Clustered Data Case (Accelerating many training snapshots, hyperparameter optimization)
#   --big-clustered-data \

#for BS in $BATS
for LR in $LRS
#for WID in $WIDS
#for MIDLYS in $MIDLYSS
do

    ${EXE} -np ${GPUS} ${PYT} ${DRVR} \
        --feat_start ${FTST} \
	--feat_stop ${FTSP} \
	--dataset ${DS} \
        --epochs ${EPCS} \
        --big-charm-data \
        --num-snapshots ${SSHOT} \
        --offset-snapshot ${SNPOFF} \
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
        --no-coords \
        --num-data-workers ${NUMW} \
        --num-test-workers ${NUMTW} \
        --calc-training-norm-only \
        --fp-row-scaling \
        --fp-standard-scaling \
        --ldos-norm-scaling \
        --ldos-max-only \
        2>&1 | tee ./logs/fp_ldos_synapse_${MAT}_${TMP}_${GC}gcc_${FPL}fp_${LDOSL}ldos_${NXYZ}nxyz_${ID}.log

done


#### Additional options

#        --save-training-data \

#        --fp-row-scaling \
#        --fp-standard-scaling \
#        --ldos-norm-scaling \
#        --ldos-max-only \

#        --power-spectrum-only \

#        --no-hidden-state \

#        --ae-factor ${AE} \


#        --ldos-max-only \

#        --fp-row-scaling \
#        --ldos-row-scaling \
#        --fp-standard-scaling \
#        --ldos-standard-scaling \
#        --fp-log \
#        --ldos-log \









