#
# Generate SNAP descriptors (fingerprints) from atomic congiurations using LAMMPS
#

import ase
import ase.io
import argparse

import numpy as np
#import io
import timeit

from lammps import lammps
import lammps_utils

from ase.calculators.lammpsrun import LAMMPS
from mpi4py import MPI

comm = MPI.COMM_WORLD

rank  = comm.Get_rank()
ranks = comm.Get_size()

print("Proc %d out of %d procs" % (rank, ranks), flush=True)

comm.Barrier()

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--temp', type=str, default="300K", metavar='S',
                            help='config temperature (default: 300K)')
parser.add_argument('--gcc', type=str, default="4.0", metavar='S',
                            help='config density in gcc (default: 4.0)')
parser.add_argument('--nxyz', type=int, default=20, metavar='N',
                            help='byn ekenebts akibg x,y,z dims (default: 20)')
args = parser.parse_args()


#temps = ["300K", "10000K", "20000K", "30000K"]

#temps = ["300K"]

#gccs = ["0.1", "0.2", "0.4", "0.6", "0.8", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"]

# Missing gcc = [0.1, 0.2] QE out file
#gccs = ["0.4", "0.6", "0.8", "1.0", "2.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"]
#gccs = ["0.1", "0.2", "3.0"]

# Test files
#gccs = ["5.0"]

temps = [args.temp]
gccs = [args.gcc]

filedir = "./%s/" % (temps[0])


echo_to_screen = False

#write_to_lammps_file = True

qe_fname = "QE_Al.scf.pw.out"
lammps_fname = "Al.scf.pw.lammps"
np_fname = "Al.fingerprint"
log_fname = "lammps_fp.log"  
 
lammps_compute_grid_fname = "./in.bgrid.python"

qe_format = "espresso-out"
lammps_format = "lammps-data"

# Grid points and training data
nx = args.nxyz
#nx = 20
ny = nx
nz = ny

# Descriptor Length Formula:
# ((tjm + 2) * (tjm + 3) * (tjm + 4)) // 24 + 3
# 5:   24
# 8:   58
# 11: 116
# 15: 245

twojmax = 11

if (rank == 0):
    print("\n---Beginning the fingerprint generation---\n", flush=True)

tic = timeit.default_timer()

for g in gccs:

    gtic = timeit.default_timer()
   
    if (rank == 0):
        print("Converting file %s gcc!" % g, flush=True)

    qe_filepath = filedir + g + "gcc/" + qe_fname
    lammps_filepath = filedir + g + "gcc/" + lammps_fname

    atoms = ase.io.read(qe_filepath, format=qe_format);

    if (rank == 0):
        print(atoms, flush=True)

    # Write to LAMMPS File
#    if (write_to_lammps_file):
    ase.io.write(lammps_filepath, atoms, format=lammps_format)
    
    if (rank == 0):
        print("Wrote QE to file for %s gcc" % (g), flush=True)
#        continue

    if (echo_to_screen):
        lmp_cmdargs = ["-echo", "screen", "-log", log_fname]
    else: 
        lmp_cmdargs = ["-screen", "none", "-log", log_fname]
    
    lmp_cmdargs = lammps_utils.set_cmdlinevars(lmp_cmdargs,
        {
        "ngridx":nx,
        "ngridy":ny,
        "ngridz":nz,
        "twojmax":twojmax,
        "atom_config_fname":lammps_filepath
        }
    )

    lmp = lammps(cmdargs=lmp_cmdargs)
  
    if (rank == 0):
        print("Computing fingerprints", flush=True)

    try:
        lmp.file(lammps_compute_grid_fname)
    except lammps.LAMMPSException:
        if (rank == 0):
            print("Bad Read of %s" % (lammps_compute_grid_fname), flush=True)

    # Check atom quantities from LAMMPS 
    num_atoms = lmp.get_natoms() 

    if (rank == 0):
        print("NUM_ATOMS: %d" % (num_atoms), flush=True)

    # Set things not accessible from LAMMPS
    # First 3 cols are x, y, z, coords

    ncols0 = 3 

    # Analytical relation for fingerprint length
    ncoeff = (twojmax+2)*(twojmax+3)*(twojmax+4)
    ncoeff = ncoeff // 24 # integer division
    fp_length = ncols0+ncoeff

    # 1. From compute sna/atom
#    bptr = lmp.extract_compute("b", 1, 2) # 1 = per-atom data, 2 = array
#    print("b = ",bptr[0][0])

    # 2. From compute sna/grid

#    bgridptr = lmp.extract_compute("bgrid", 0, 2) # 0 = style global, 2 = type array
#    print("bgrid = ",bgridptr[0][ncols0+0])

    # 3. From numpy array pointing to sna/atom array
     
#    bptr_np = lammps_utils.extract_compute_np(lmp,"b",1,2,(num_atoms,ncoeff))
#    print("b_np = ",bptr_np[0][0])

    # 4. From numpy array pointing to sna/grid array

    bgridptr_np = lammps_utils.extract_compute_np(lmp, "bgrid", 0, 2, (nz,ny,nx,fp_length))

#    print("bgrid_np = ",bgridptr_np[0][0][0][ncols0+0])
    if (rank == 0):
        print("bgrid_np size = ",bgridptr_np.shape, flush=True)


    fingerprint_filepath = filedir + g + "gcc/" + np_fname
    # Save LAMMPS numpy array as binary 
    if (rank == 0):
        np.save(fingerprint_filepath, bgridptr_np, allow_pickle=True)

    gtoc = timeit.default_timer()

    comm.Barrier()

    print("Rank %d Time %s gcc: %4.4f" % (rank, g, gtoc - gtic), flush=True)

toc = timeit.default_timer()

comm.Barrier()

print("Rank %d, Total conversion time: %4.4f" % (rank, toc - tic), flush=True)

comm.Barrier()

if (rank == 0):
    print("\n\nSuccess!", flush=True)
