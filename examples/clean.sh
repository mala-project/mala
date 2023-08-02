#!/bin/sh

set -u

# Remove artifact files that some example scripts write.

here=$(dirname $(readlink -f $0))
for dir in basic advanced; do
    cd $here/$dir
    echo "cleaning: $(pwd)"
    rm -rvf \
        *.pth \
        *.pkl \
        *.db \
        *.pw* \
        __pycache__ \
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
