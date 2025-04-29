#!/bin/bash

set -ue

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


# -lfoo will link shared libs (libfoo.so) as well as static ones (libfoo.a). In
# qe_libs we list the ones we need in total_energy (which are almost all of
# them anyway), but since we are lazy, we use all paths to them as link (-L)
# and include (-I) dirs.
qe_static_lib_dirs=$(find $root_dir -name "*.a" | xargs dirname | sort -u | paste -s -d '@')
qe_lib_dirs=-L$(echo "$qe_static_lib_dirs" | sed -re 's/@/ -L/g')
qe_inc_dirs=-I$(echo "$qe_static_lib_dirs" | sed -re 's/@/ -I/g')
qe_libs="-lqemod -lks_solvers -lqefft -lqela -lutil -ldftd3qe -lupf -ldevXlib -lmbd"

echo "qe_lib_dirs: $qe_lib_dirs"
echo "qe_inc_dirs: $qe_inc_dirs"

# default: system blas,lapack and fftw, adapt as needed
linalg="-lblas -llapack"
fftw="-lfftw3"

here=$(pwd)
mod_name=total_energy
mkdir -p $here/libs

# xc_lib.a needs to be called lib<someting>.a to be linkable with -l<something>
cp $root_dir/XClib/xc_lib.a $here/libs/libxclib.a

# Stick all object files into a static lib so that we can link them, just
# specifying them on the f2py CLI doesn't work with the meson backend, as it
# seems.
ar rcs $here/libs/liballobjs.a $pwobjs $utilobjs $modobjs

FFLAGS="-I$here/libs -I$pw_src_path $qe_inc_dirs" \
LDFLAGS="-L$here/libs -L$pw_src_path $qe_lib_dirs" \
FC="mpif90" \
python3 -m numpy.f2py --backend meson \
    -c ${mod_name}.f90 \
    -m $mod_name \
    $linalg \
    $fftw \
    $qe_libs \
    -lallobjs -lxclib
