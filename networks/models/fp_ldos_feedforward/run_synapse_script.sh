
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
    echo "Default 1 GPU"
    GPUS=1
fi

#DS=random
DS=fp_ldos
#EPCS=100
EPCS=2
MDL=5
NXYZ=200
TM=298K
GC=2.699
SSHOT=5
LR=.01
ES=.999999


${EXE} -np ${GPUS} ${PYT} ${DRVR} --dataset ${DS} --epochs ${EPCS} --model ${MDL} --nxyz ${NXYZ} --temp ${TM} --gcc ${GC} --early-stopping ${ES} --lr ${LR} --num-snapshots ${SSHOT} 2>&1 | tee ./logs/fp_ldos_synapse_${GPUS}gpus_${TM}_${GC}gcc_${MDL}model_${SSHOT}snapshots_${EPCS}epochs_${NXYZ}nxyz_${LR}learnrate.log

