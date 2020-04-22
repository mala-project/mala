
EXE="horovodrun --gloo-timeout-seconds 600"
#EXE="mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
#EXE=mpirun
#EXE=mpiexec

PYT=python3
DRVR=temp_fp_ldos_src.py


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
#EPCS=250
EPCS=1
#MDL=4
#MDL=6
MDL=5
NXYZ=200
TM=298K
GC=2.699

BS=64
#BS=4

#FPL="8 17 33 58"
#FPL="58"
#FPL=8
#FPL=17
#FPL=33
#FPL=58
FPL=94
LDOSL=128
#SSHOT=5
SSHOT=3
LR=.01
ES=.99999
EP=8
OP=4

WID=300
MIDLYS=4
AE=.8


for FL in $FPL
do

    ${EXE} -np ${GPUS} ${PYT} ${DRVR} \
        --dataset ${DS} \
        --epochs ${EPCS} \
        --model-lstm-network \
        --nxyz ${NXYZ} \
        --temp ${TM} \
        --gcc ${GC} \
        --batch-size ${BS} \
        --fp-length ${FL} \
        --ldos-length ${LDOSL} \
        --optim-patience ${OP} \
        --early-patience ${EP} \
        --early-stopping ${ES} \
        --lr ${LR} \
        --ff-mid-layers ${MIDLYS} \
        --ff-width ${WID} \
        --ae-factor ${AE} \
        --num-snapshots ${SSHOT} \
        --no-coords \
        --no-hidden-state \
        --big-data \
        2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${FL}fp_${LDOSL}ldos_${NXYZ}nxyz_nocoords_LSTM_${BS}batchsize_${SSHOT}snapshots_${EPCS}epochs_${LR}learnrate_fprow_fpstand_ldosmax.log

done


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









