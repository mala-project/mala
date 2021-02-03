#!/bin/bash

echo "In addition to sourcing this script, source mlmm-ldrd-data/networks/environment/load_mlmm_blake_modules.sh"

export FCFLAGS="-O3 -g -fPIC"
export FFLAGS="-O3 -g -fPIC"
export CFLAGS=-fPIC
export CPPFLAGS=-fPIC
