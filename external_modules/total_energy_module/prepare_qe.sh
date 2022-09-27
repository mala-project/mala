#!/bin/bash

while getopts q: flag
do
    case "${flag}" in
        q) root_dir=${OPTARG};;
    esac
done
if [ ! "$root_dir" ]; then
  echo "Error: no path to QE provided (missing argument -q)"
  exit 1
fi

cd $root_dir
if [ "$(grep -q "$foxflags -fPIC" install/configure)" == "" ]; then
  sed -i 's/# Checking preprocessor.../foxflags="$foxflags -fPIC"\n# Checking preprocessor.../g' install/configure
fi
