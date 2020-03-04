
EXE=horovodrun
#EXE="mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
#EXE=mpirun
#EXE=mpiexec

PYT=python3
DRVR=fp_ldos_ff_driver.py


if [ $1 ]
then
    echo "$1 GPUS"
    GPUS=$1
else
    echo "Default 2 GPUs"
    GPUS=2
fi

#DS=random
DS=fp_ldos
#EPCS=100
EPCS=1
MDL=5
NXYZ=200
TM=298K
GC=2.699
#SSHOT=5
SSHOT=3
LR=.000001
ES=.999
EP=2
OP=1



${EXE} -np ${GPUS} ${PYT} ${DRVR} \
    --dataset ${DS} \
    --epochs ${EPCS} \
    --model ${MDL} \
    --nxyz ${NXYZ} \
    --temp ${TM} \
    --gcc ${GC} \
    --optim-patience ${OP} \
    --early-patience ${EP} \
    --early-stopping ${ES} \
    --lr ${LR} \
    --num-snapshots ${SSHOT} \
    --fp-row-scaling \
    --fp-standard-scaling \
    --ldos-max-only \
    2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate_fprow_fpstand_ldosrow_ldosstand.log






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









