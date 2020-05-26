
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
    echo "Default 3 GPUs"
    GPUS=3
fi


### Input/Output

#DS=random
DS=fp_ldos

MAT=H2O
NXYZ=75
TMP=300K
GC=1.0

FPL=94
LDOSL=1

#SSHOT=5
#SSHOT=3

SSHOT=10



### Optimizer/Training

EPCS=250
#EPCS=100
#EPCS=1


# Test Batch Size (TBS) and Training Batch Size (BS) 
# GPUS * BS must evenly divide NUM_GRIDPTS = 75^3 = 41875
TBS=3125
#BS=3125
#BS=625
BS=100


# Learning rates
#LRS=".0005 .0001 .00005 .00001"
LRS=".0001 .00005 .00001 .000005 .000001"
#LR=.0001 

# Necessary reduction in validation loss 
# (must be less that best_loss * ${ES})
ES=.99999

# After 8 consecutive failures end training.
EP=8

# After 4 consecutive failures reduce learning rate by factor of 10
OP=4

#LSTMIL="1 2 4 8"


# Last output layer size multiple.
# I.E. if output size = 1 (density-case)
# then next to last layer is size ${LIL}
# if ouptut size = 250 (ldos-case)
# then next to last layer is size ${LIL} * 250
LIL=20

# Asynch GPU data-loading
NUMW=2

# Network

#WIDS="20 50 100 200 300 400"
WID=20

#MIDLYSS="1 2 3 4 5 6"
MIDLYS=2

#MIDLYS=2
AE=.8

#   --model-lstm-network \
#   --adam \
#   --skip-connection \


for LR in $LRS
do

    ${EXE} -np ${GPUS} ${PYT} ${DRVR} \
        --dataset ${DS} \
        --material ${MAT} \
        --epochs ${EPCS} \
        --adam \
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
        --calc-training-norm-only \
        --fp-row-scaling \
        --fp-standard-scaling \
        --ldos-norm-scaling \
        --ldos-max-only \
        2>&1 | tee ./logs/fp_ldos_synapse_${MAT}_${GPUS}gpus_${TMP}_${GC}gcc_${FPL}fp_${LDOSL}ldos_${NXYZ}nxyz_${RANDOM}${RANDOM}.log

done




#        --save-training-data \

#        --fp-row-scaling \
#        --fp-standard-scaling \
#        --ldos-norm-scaling \
#        --ldos-max-only \

#        --power-spectrum-only \

#    --no-hidden-state \

#    --fp-row-scaling \
#    --fp-standard-scaling \
#    --ldos-max-only \




############ FP LDOS Row
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-row-scaling \
#    --ldos-row-scaling \
#    --fp-standard-scaling \
#    --ldos-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpstand_ldosrow_ldosstand.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-row-scaling \
#    --ldos-row-scaling \
#    --fp-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpstand_ldosrow_ldosmm.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-row-scaling \
#    --ldos-row-scaling \
#    --ldos-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpmm_ldosrow_ldosstand.log
#
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-row-scaling \
#    --ldos-row-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpmm_ldosrow_ldosmm.log
#
############ FP Row, LDOS Total
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-row-scaling \
#    --fp-standard-scaling \
#    --ldos-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpstand_ldostot_ldosstand.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-row-scaling \
#    --fp-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpstand_ldostot_ldosmm.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-row-scaling \
#    --ldos-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpmm_ldostot_ldosstand.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-row-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpmm_ldostot_ldosmm.log
#
#
############ FP Tot, LDOS Row
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --ldos-row-scaling \
#    --fp-standard-scaling \
#    --ldos-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fptot_fpstand_ldosrow_ldosstand.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --ldos-row-scaling \
#    --fp-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fptot_fpstand_ldosrow_ldosmm.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --ldos-row-scaling \
#    --ldos-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fptot_fpmm_ldosrow_ldosstand.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --ldos-row-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fptot_fpmm_ldosrow_ldosmm.log
#
#
#
############ FP LDOS Tot
#
#
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-standard-scaling \
#    --ldos-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fptot_fpstand_ldostot_ldosstand.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --fp-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fptot_fpstand_ldostot_ldosmm.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    --ldos-standard-scaling \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fptot_fpmm_ldostot_ldosstand.log
#
#
#${EXE} -np ${GPUS} ${PYT} ${DRVR} \
#    --dataset ${DS} \
#    --epochs ${EPCS} \
#    --model ${MDL} \
#    --nxyz ${NXYZ} \
#    --temp ${TM} \
#    --gcc ${GC} \
#    --optim-patience ${OP} \
#    --early-patience ${EP} \
#    --early-stopping ${ES} \
#    --lr ${LR} \
#    --num-snapshots ${SSHOT} \
#    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fptot_fpmm_ldostot_ldosmm.log
#
#




#    --fp-row-scaling \
#    --ldos-row-scaling \
#    --fp-standard-scaling \
#    --ldos-standard-scaling \
#    --fp-log \
#    --ldos-log \









