#!/bin/bash

while getopts q: flag
do
    case "${flag}" in
        q) root_dir=${OPTARG};;
    esac
done

cd $root_dir
if [ "$(grep -q "$foxflags -fPIC" install/configure)" == "" ]; then
  echo "HERE"
  sed -i 's/# Checking preprocessor.../foxflags="$foxflags -fPIC"\n# Checking preprocessor.../g' install/configure
fi
