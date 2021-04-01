

WORK_DIR=/ascldap/users/johelli/Code/mlmm/mlmm-ldrd-data/
LAMMPS_DIR=${WORK_DIR}/networks/lammps
BUILD_DIR=${WORK_DIR}/build_lammps

source ${WORK_DIR}/networks/load_mlmm_blake_modules.sh

cd ${BUILD_DIR}

cmake \
-D PKG_SNAP:BOOL=ON \
-D LAMMPS_EXCEPTIONS:BOOL=ON \
-D BUILD_LIB:BOOL=ON \
-D BUILD_SHARED_LIBS:BOOL=ON \
${LAMMPS_DIR}/cmake/ |& tee OUTPUT.CMAKE 

make -j8 |& tee OUTPUT.MAKE

