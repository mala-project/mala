#!/bin/bash

# usage:
#
#   $ ./this.sh
# or
#   $ F2PY=/usr/bin/f2py3 ./this.sh

set -u

err(){
    echo "error $@"
    exit 1
}

[ $# -eq 1 ] || err "Please provide exactly one argument (the path to the QE directory)" && root_dir=$1

echo "Using QE root dir: $root_dir"

pw_src_path=$root_dir/PW/src

# Object files from the PW/src folder
pwobjs="$(find $pw_src_path -name "*.o" | grep -E -v 'generate_vdW_kernel_table|generate_rVV10_kernel_table')"

# Object files from the UtilXlib folder
utilobjs="$(find $root_dir/UtilXlib/ -name "*.o")"

# Object files from the Modules folder
modobjs="$(find $root_dir/Modules/ -name "*.o")"

[ -n "$pwobjs" ] || err "pwobjs empty"
[ -n "$utilobjs" ] || err "utilobjs empty"
[ -n "$modobjs" ] || err "modobjs empty"

project_lib_folders=" -L$root_dir/Modules -L$root_dir/KS_Solvers -L$root_dir/FFTXlib/src -L$root_dir/LAXlib -L$root_dir/UtilXlib -L$root_dir/dft-d3 -L$root_dir/upflib -L$root_dir/XClib -L$root_dir/external/devxlib/src -L$root_dir/external/mbd/src"
project_libs="-lqemod -lks_solvers -lqefft -lqela -lutil -ldftd3qe -lupf -ldevXlib -lmbd"
project_inc_folders="-I$root_dir/Modules -I$root_dir/FFTXlib/src -I$root_dir/LAXlib -I$root_dir/KS_Solvers -I$root_dir/UtilXlib -I$root_dir/upflib -I$root_dir/XClib -I$root_dir/external/devxlib/src -I$root_dir/external/mbd/src"

# default: system blas,lapack and fftw, adapt as needed
linalg="-lblas -llapack"
fftw="-lfftw3"


here=$(pwd)
mod_name=total_energy
src=${mod_name}.f90
cp -v $src $pw_src_path/
cd $pw_src_path
rm -vf ${mod_name}.*.so

${F2PY:=f2py} \
    --f90exec=mpif90 \
    --f77exec=mpif90 \
    -c $src \
    -m $mod_name \
    $project_inc_folders \
    $project_lib_folders \
    $project_libs \
    $pwobjs \
    $utilobjs \
    $modobjs \
    $linalg \
    $fftw \
    $root_dir/XClib/xc_lib.a

rm $src
mv -v ${mod_name}.*.so $here/
cd $here
