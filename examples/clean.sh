#!/bin/sh

# Remove artifact files that some example scripts write.

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
    lammps*.tmp
