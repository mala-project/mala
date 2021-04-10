#!/bin/bash

echo "Setting environment variables for total energy module compilation."

export FCFLAGS="-O3 -g -fPIC"
export FFLAGS="-O3 -g -fPIC"
export CFLAGS=-fPIC
export CPPFLAGS=-fPIC
