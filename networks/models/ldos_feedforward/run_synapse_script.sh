
EXE=horovodrun
#EXE="mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib"
#EXE=mpirun
#EXE=mpiexec

GPUS=2

DS=random
EPCS=100
MDL=5
NXYZ=20
TM=298K
GC=2.699



${EXE} -np ${GPUS} python3 ldos_ff_network.py --dataset ${DS} --epochs ${EPCS} --model ${MDL} --nxyz ${NXYZ} --temp ${TM} --gcc ${GC}

