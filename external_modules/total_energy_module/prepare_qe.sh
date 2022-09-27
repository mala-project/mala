#!/bin/bash

set -euo pipefail

err(){
    echo "error $@"
    exit 1
}

[ $# -eq 1 ] || err "Please provide exactly one argument (the path to the QE directory)" && root_dir=$1

cd $root_dir
if [ "$(grep -q "\$foxflags -fPIC" install/configure)" == "" ]; then
  sed -i 's/# Checking preprocessor.../foxflags="$foxflags -fPIC"\n# Checking preprocessor.../g' install/configure
fi
