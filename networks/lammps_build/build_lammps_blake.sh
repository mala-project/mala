

WORK_DIR=/ascldap/users/johelli/Code/mlmm/mlmm-ldrd-data/
LAMMPS_DIR=${WORK_DIR}/networks/lammps/
BUILD_DIR=${WORK_DIR}/lammps_build

source ${WORK_DIR}/networks/load_mlmm_blake_modules.sh

cmake \
-D PKG_SNAP=ON \
-D LAMMPS_EXCEPTIONS=yes \
${LAMMPS_DIR}/cmake/ |& tee OUTPUT.CMAKE 

make -j8 |& tee OUTPUT.MAKE

