#!/bin/bash


# Parse user arguments.
while getopts ":v:p:" opt; do
  case $opt in
    v) qe_version="$OPTARG"
    ;;
    p) path_to_qe="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Find out where the script is located.
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Check user arguments.
case $qe_version in
  "6.6")
    echo "This script will install the total energy module for QE version"\
    $qe_version", located at "$path_to_qe"/"
  ;;

  "6.4.1")
    echo "This script will install the total energy module for QE version"\
    $qe_version", located at "$path_to_qe"/"
  ;;

  *)
    echo "Invalid QE version selected. Currently only 6.4.1 and 6.6 are supported."
    exit
  ;;
esac


# Set everything up.
source tem_set_environment.sh
source tem_set_objects.sh $qe_version

# Move to QE and copy the total energy module
echo "Copying source code to QE installation directory."
cd $path_to_qe/PW/src
case $qe_version in
  "6.6")
    cp $script_path/_qe6_6.total_energy.f90 total_energy.f90
  ;;

  "6.4.1")
    cp $script_path/_qe6_4_1.total_energy.f90 total_energy.f90
  ;;
esac

f2py --f90exec=mpif90 --fcompiler=gnu95 -c total_energy.f90 -m total_energy $PROJECT_INC_FOLDERS $PROJECT_LIB_FOLDERS \
     $PROJECT_LIBS $PWOBJS $UTILOBJS $MODOBJS $UPFOBJS $BEEFOBS $LINALGOBJS $LINALG $FOXLIBS ../../clib/clib.a
