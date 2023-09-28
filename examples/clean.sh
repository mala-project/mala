#!/bin/sh

# Remove artifact files that some example scripts write.

cd basic
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
cd ..
cd advanced
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
cd ..
