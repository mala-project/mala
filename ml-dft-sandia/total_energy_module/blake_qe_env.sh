#!/bin/bash

module purge
module load python/3.7.3

#. f2py_env/bin/activate

module load intel/compilers/18.1.163
module load openmpi/2.1.2/intel/18.1.163

export FCFLAGS=-fPIC
export FFLAGS=-fPIC
export CFLAGS=-fPIC
export CPPFLAGS=-fPIC

