#
# Generate SNAP descriptors (fingerprints) from atomic congiurations using LAMMPS
#

import ase
import ase.io
import argparse
import os, sys
import itertools

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

if (rank == 0):
    print("\n-----------------------------------\n")
    print("----  GENERATE LAMMPS FP DATA  ----")
    print("\n-----------------------------------\n")

parser = argparse.ArgumentParser(description='Fingerprint Generation')

parser.add_argument('--water', action='store_true', default=False,
                            help='run fingerprint generation for water files')
parser.add_argument('--atom-based', action='store_true', default=False,
                            help='run lammps fingerprints for atoms instead of grid files')
parser.add_argument('--run-all', action='store_true', default=False,
                            help='run ldos parser for all densities and temperatures')
parser.add_argument('--temp', type=str, default="298K", metavar='S',
                            help='temperature of fingerprint in units K (default: "298K")')
parser.add_argument('--gcc', type=str, default="2.699", metavar='S',
                            help='density of fingerprint in units in g/cm^3 (default: "2.699")')
parser.add_argument('--snapshot', type=str, default="0", metavar='S',
                            help='snapshot number at given gcc/temp (default: "0")')
parser.add_argument('--nxyz', type=int, default=20, metavar='N',
                            help='number of grid cells in the X/Y/Z dimensions (default: 20)')
parser.add_argument('--rcutfac', type=float, default=4.67637, metavar='R',
                            help='radius cutoff factor for the fingerprint sphere in Angstroms (default: 4.67637)')
parser.add_argument('--twojmax', type=int, default=10, metavar='N',
                            help='band limit for fingerprints (default: 10)')
parser.add_argument('--no-qe', action='store_true', default=False,
                            help='use LAMMPS input file directly (default: False)')

parser.add_argument('--data-dir', type=str, \
                default="../../fp_data", \
                metavar="str", help='path to data directory with QE output files (default: ../../fp_data)')
parser.add_argument('--output-dir', type=str, default="../../fp_data",
                metavar="str", help='path to output directory (default: ../../fp_data)')
args = parser.parse_args()

# Print arguments
print("Parser Arguments")
for arg in vars(args):
        print ("%s: %s" % (arg, getattr(args, arg)))

if (args.water):
    print("Using this script to generate fingerprints for water")

    #args.data_dir = '/ascldap/users/jracker/water64cp2k/datast_1593/results/'

    temp_grid = np.array([args.temp])
    gcc_grid = np.array(['aaaa'])
    #gcc_grid = [''.join(i) for i in itertools.product("abcdefghijklmnopqrstuvwxyz",repeat=4)][:1593]
    snapshot_grid = np.array([args.snapshot])

    cube_filename_head = "w64_"
    cube_filename_tail = "-ELECTRON_DENSITY-1_0.cube"


    #args.atom_based = True

elif (args.run_all):

    # Currently available
    temp_grid = np.array(["298K"])
    # Future
#    temp_grid = np.array(["300K", "10000K", "20000K", "30000K"])


    # Currently available
    gcc_grid = np.array(["2.699"])
#    gcc_grid = np.array(["0.4", "0.6", "0.8", "1.0", "2.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"])
    
    # Future
    # Missing gcc = [0.1, 0.2] QE out file
#    gcc_grid = np.array(["0.1", "0.2", "0.4", "0.6", "0.8", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"])
    # Test
#    gcc_grid = np.array(["2.0", "3.0"])

else:

    temp_grid = np.array([args.temp])
    gcc_grid = np.array([args.gcc])

echo_to_screen = False

#write_to_lammps_file = True

qe_fname = "QE_Al.scf.pw.snapshot%s.out" % args.snapshot
lammps_fname = "Al.scf.pw.snapshot%s.lammps" % args.snapshot
log_fname = "lammps_fp_%dx%dx%dgrid_%dtwojmax_snapshot%s.log" % \
        (args.nxyz, args.nxyz, args.nxyz, args.twojmax, args.snapshot) 


# First 3 cols are x, y, z, coords

ncols0 = 3 

# Analytical relation for fingerprint length
ncoeff = (args.twojmax+2)*(args.twojmax+3)*(args.twojmax+4)
ncoeff = ncoeff // 24 # integer division
fp_length = ncols0 + ncoeff


if (args.water):
    qe_format = "cube"
    np_fname = "water_fp_%dx%dx%dgrid_%dcomps_snapshot%s" % args.snapshot
    lammps_compute_grid_fname = "./in.bgrid.twoelements.python"
else:
    qe_format = "espresso-out"
    np_fname = "Al_fp_%dx%dx%dgrid_%dcomps_snapshot%s" % (args.nxyz, args.nxyz, args.nxyz, fp_length, args.snapshot)
    lammps_compute_grid_fname = "./in.bgrid.python"

lammps_format = "lammps-data"

# Grid points and training data
nx = args.nxyz
ny = nx
nz = ny

# Descriptor Length Formula:
# ((tjm + 2) * (tjm + 3) * (tjm + 4)) // 24 + 3
# 5:   24
# 8:   58
# 11: 116
# 15: 245

tic = timeit.default_timer()

# Loops over temperature grid
for temp in temp_grid:

    print("\nWorking on Temp %s" % temp)

    # Loop over density grid
    for gcc in gcc_grid:

        inner_tic = timeit.default_timer()
      
        temp_dir =  args.data_dir + "/%s" % (temp)
        gcc_dir = temp_dir + "/%sgcc/" % (gcc)

        if (args.water):
            qe_filepath = args.data_dir + cube_filename_head + gcc + cube_filename_tail
            lammps_filepath = args.data_dir + cube_filename_head + gcc + ".lammps" 
        else:
            # Make Temp directory
            if not os.path.exists(temp_dir):
                print("\nWarning! Creating input folder %s" % temp_dir)
                os.makedirs(temp_dir)
            # Make Density directory
            if not os.path.exists(gcc_dir):
                print("\nWarning! Creating input folder %s" % gcc_dir)
                os.makedirs(gcc_dir)
   
            qe_filepath = gcc_dir + qe_fname
            lammps_filepath = gcc_dir + lammps_fname
    
        if (args.no_qe):
            print("Skipping QE conversion. Reading LAMMPS input directly.")
        elif not os.path.exists(qe_filepath):
            print("\n\nQE out file %s does not exist! Exiting.\n\n" % (qe_filepath))
            exit(0)
        
        if (args.no_qe):
            if (not os.path.exists(lammps_filepath)):
                print("\n\nLAMMPS file %s does not exist! Exiting.\n\n" % (lammps_filepath))
                exit(0)
            
        elif (not os.path.exists(lammps_filepath) and not args.no_qe):
        
            if (rank == 0):
                print("\nConverting %sgcc file %s to %s!" % (gcc, qe_filepath, lammps_filepath), flush=True)
            
            atoms = ase.io.read(qe_filepath, format=qe_format);

            if (rank == 0):
                print(atoms, flush=True)

            # Write to LAMMPS File
            ase.io.write(lammps_filepath, atoms, format=lammps_format)
        
            if (rank == 0):
                print("Wrote QE to file for %s gcc" % (gcc), flush=True)
        else:
            print("\nLAMMPS input file %s already exists.\n" % lammps_filepath)

        if (echo_to_screen):
            lmp_cmdargs = ["-echo", "screen", "-log", log_fname]
        else: 
            lmp_cmdargs = ["-screen", "none", "-log", log_fname]
        
        lmp_cmdargs = lammps_utils.set_cmdlinevars(lmp_cmdargs,
            {
            "ngridx":nx,
            "ngridy":ny,
            "ngridz":nz,
            "twojmax":args.twojmax,
            "rcutfac":args.rcutfac,
            "atom_config_fname":lammps_filepath
            }
        )

        lmp = lammps(cmdargs=lmp_cmdargs)
      
        if (rank == 0):
            print("\nComputing fingerprints...", flush=True)

        try:
            lmp.file(lammps_compute_grid_fname)
        except lammps.LAMMPSException:
            if (rank == 0):
                print("Bad Read of %s" % (lammps_compute_grid_fname), flush=True)

        # Check atom quantities from LAMMPS 
        num_atoms = lmp.get_natoms() 

        if (rank == 0):
            print("TEST, NUM_ATOMS: %d" % (num_atoms), flush=True)

        # Extract numpy array pointing to sna/atom array

        if (args.atom_based):
            print("\n\nFor Josh\n\n")
            exit(0)

            bptr_np = lammps_utils.extract_compute_np(lmp, "b", 1, 2, (num_atoms,ncoeff))


        # Extract numpy array pointing to sna/grid array (Z, Y, X ordering)
        else:
            bptr_np = lammps_utils.extract_compute_np(lmp, "bgrid", 0, 2, (nz,ny,nx,fp_length))

            # switch from x-fastest to z-fastest order (swaps 0th and 2nd dimension)
            bptr_np = bptr_np.transpose([2,1,0,3])

        if (rank == 0):
            print("bptr_np shape = ",bptr_np.shape, flush=True)

        # Output location
        temp_dir = args.output_dir + "/%s" % temp
        gcc_dir = temp_dir + "/%sgcc/" % gcc 

        # Make Temp directory
        if not os.path.exists(temp_dir):
            print("\nCreating output folder %s" % temp_dir)
            os.makedirs(temp_dir)
        # Make Density directory
        if not os.path.exists(gcc_dir):
            print("\nCreating output folder %s" % gcc_dir)
            os.makedirs(gcc_dir)

        fingerprint_filepath = gcc_dir + np_fname
        # Save LAMMPS numpy array as binary 
        if (rank == 0):
            np.save(fingerprint_filepath, bptr_np, allow_pickle=True)

        inner_toc = timeit.default_timer()

        comm.Barrier()

        print("Rank %d Time %s gcc: %4.4f" % (rank, gcc, inner_toc - inner_tic), flush=True)

toc = timeit.default_timer()

comm.Barrier()

print("Rank %d, Total conversion time: %4.4f" % (rank, toc - tic), flush=True)

comm.Barrier()

if (rank == 0):
    print("\n\nSuccess!", flush=True)
