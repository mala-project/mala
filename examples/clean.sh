#!/bin/sh

set -u

# Remove artifact files that some example scripts write.

here=$(dirname $(readlink -f $0))
for dir in $here $(find $here -mindepth 1 -type d | grep -vE '\.ruff_cache|__pycache__'); do
    cd $dir
    echo "cleaning: $(pwd)"
    rm -rvf \
        *.pth \
        *.pkl \
        *.pk \
        *.db \
        *.pw* \
        *.cube \
        ex10_vis \
        *.tmp \
        *.npy \
        *.json \
        *.zip \
        Be_snapshot* \
        lammps*.tmp \
        mala_vis
done
