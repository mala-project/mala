#!/bin/bash                                                 


# Run `python3 ldos_example.py --help` for more option information

NODES=2
RPN=19
RANKS=$((${NODES}*${RPN}));

echo "Total ranks: ${RANKS}"

EXE="mpirun -np ${RANKS}" 

TEMPS="298K" 

#GCCS="0.4 0.6 0.8 1.0 2.0 3.0 4.0 6.0 7.0 8.0 9.0"
FILEPATH="/home/jracker/qmdata/w64_data/cube_data/*.cube"


echo $(date)

for T in $TEMPS
do
    for G in $FILEPATH
    do
        PREFIX="/home/jracker/qmdata/w64_data/cube_data/w64_"
        GG=${G%-ELECTRON_DENSITY-1_0.cube}
        GGG=${GG#$PREFIX}
        echo "Running fingerprint generation for temp ${T} and gcc ${GGG}" 

        #check if this file exists, if so skip!


        ${EXE} python generate_fingerprints.py --water --gcc ${GGG} --twojmax 10 --nxyz 75 --data-dir ~/qmdata/w64_data/cube_data/ --output-dir ~/qmdata/w64_data/fp_twojmax_10/ 

        #${EXE} python3 generate_fingerprints.py --temp ${T} --gcc ${G} --nxyz ${NXYZ} > logs/gen_fp_n${NODES}_ranks${RANKS}_temp${T}_gcc${G}_nxyz${NXYZ}.log
    done
done

echo $(date) 

exit 0
