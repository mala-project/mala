import os
import sys
import numpy as np
import timeit
import itertools
import argparse

import cube_parser

print("\n-----------------------------------\n")
print("--------  PARSE LDOS DATA  --------")
print("\n-----------------------------------\n")

parser = argparse.ArgumentParser(description='LDOS Parser')

# Targets
parser.add_argument('--water', action='store_true', default=False,
                            help='run ldos parser for water files')
parser.add_argument('--run-all-water', action='store_true', default=False,
                            help='run ldos parser for all densities and temperatures')
parser.add_argument('--density', action='store_true', default=False,
                            help='run density parser for each density and temperature, also')

# Dimensions
parser.add_argument('--nxyz', type=int, default=200, metavar='N',
                            help='number of grid cells in the X/Y/Z dimensions (default: 200)')
parser.add_argument('--elvls', type=int, default=250, metavar='E',
                            help='the number of energy levels in the LDOS (for density only e=1) (default: 250)')

# Directories
parser.add_argument('--material', type=str, default="Al", metavar='T',
                            help='material of ldos parser (default: "Al")')
parser.add_argument('--temp', type=str, default="298K", metavar='T',
                            help='temperature of ldos parser in units K (default: 298K)')
parser.add_argument('--gcc', type=str, default="2.699", metavar='GCC',
                            help='density of ldos parser in units g/cm^3 (default: "2.699")')
parser.add_argument('--snapshot', type=str, default="0", metavar='T',
                            help='snapshot for ldos parser (default: 0)')
parser.add_argument('--cube-fname-head', type=str, default="/tmp.pp", metavar='H',
                            help='head of the ldos cube filenames (default: "/tmp.pp")')
parser.add_argument('--cube-fname-tail', type=str, default="_ldos.cube", metavar='T',
                            help='tail of the ldos cube filenames (default: "_ldos.cube")')
parser.add_argument('--density-cube-fname', type=str, default="Al_dens.cube", metavar='C',
                            help='density cube filenames (default: "Al_dens.cube")')

parser.add_argument('--data-dir', type=str, \
        default="./cube_files", \
        metavar="str", help='path to data directory (default: "./cube_files")')

parser.add_argument('--output-dir', type=str, default="../../ldos_data",
        metavar="str", help='path to output directory (default: ../ldos_data)')

args = parser.parse_args()


if (args.water):
    print ("Using this script in water density reading mode")
    #args.data_dir = '/ascldap/users/jracker/water64cp2k/datast_1593/results/'
    #args.data_dir = '/Users/jracker/ldrd_ml_density/mlmm-ldrd-data/water/'

    temp_grid = np.array([args.temp])
    gcc_grid = np.array(['aaaa'])
    snapshot_grid = np.array([args.snapshot])

    if (args.run_all_water):
        gcc_grid = [''.join(i) for i in itertools.product("abcdefghijklmnopqrstuvwxyz",repeat=4)][:1593]
    
    print (gcc_grid)

    #cube_filename_head = "w64_"
    #cube_filename_tail = "-ELECTRON_DENSITY-1_0.cube"


    # only doing density: 1 level
    #e_lvls = args.elvls
    e_lvls = 1

else:
  
    # 1 snapshot per parse_ldos call, for now
    temp_grid = np.array([args.temp])
    gcc_grid = np.array([args.gcc])
    snapshot_grid = np.array([args.snapshot])


    # Energy levels for LDOS
    e_lvls = args.elvls

# Output location of .npy binary files
#output_dir = '../ldos_data'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Output filename head
npy_filename_head = "/%s_ldos_" % args.material
npy_filename_head_dens = "/%s_dens_" % args.material

xyze_shape = [args.nxyz, args.nxyz, args.nxyz, args.elvls]
dens_shape = [args.nxyz, args.nxyz, args.nxyz]

tot_tic = timeit.default_timer()

# Print arguments
print("Parser Arguments")
for arg in vars(args):
    print ("%s: %s" % (arg, getattr(args, arg)))

# Loop over temperature grid
for temp in temp_grid:

    print("\nWorking on Temp %s" % temp)

    # Loop over density grid
    for gcc in gcc_grid:

        inner_tic = timeit.default_timer()
        
        temp_dir =  args.output_dir + "/%s" % (temp)
        gcc_dir = temp_dir + "/%sgcc" % (gcc)

        out_filename = gcc_dir + npy_filename_head + "%dx%dx%dgrid_%delvls_snapshot%s" % \
                (args.nxyz, args.nxyz, args.nxyz, args.elvls, args.snapshot) 

        if (args.water):
            out_filename = args.data_dir + args.cube_fname_head + gcc + args.cube_fname_tail + "_pkl"
        else:
            # Make Temperature directory
            if not os.path.exists(temp_dir):
                print("\nCreating folder %s" % temp_dir)
                os.makedirs(temp_dir)
            # Make Density directory
            if not os.path.exists(gcc_dir):
                print("\nCreating folder %s" % gcc_dir)
                os.makedirs(gcc_dir)

        # Allocate space
        ldos = np.empty(xyze_shape)

        # Loop over energy grid
        # separate cube file for each ldos 
        for e in range(e_lvls):
#            infile_name = args.data_dir + "/%sgcc/" % (gcc) + \
#            infile_name = args.data_dir + "/%s" % temp + "/%sg/" % (gcc) + \
#                          cube_fname_head + "%d" % (e) + cube_fname_tail

            infile_name = args.data_dir + args.cube_fname_head + \
                    str(e + 1).zfill(len(str(e_lvls))) + args.cube_fname_tail

            if (args.water):
                infile_name = args.data_dir + args.cube_fname_head + gcc + "-" + args.cube_fname_tail  

            print("\nLoading data from %s" % infile_name)

            # load into numpy array (ignore metadata w/ '_')
            ldos[:,:,:,e], _ = cube_parser.read_cube(infile_name)


        # Save numpy array to binary file
        print("Saving LDOS to %s" % (out_filename))
        np.save(out_filename, ldos, allow_pickle=True)


        # If parsing density, as well
        if (args.density):
            infile_name = args.data_dir + "/%sgcc/" % (gcc) + args.density_cube_fname 
            outfile_name = args.output_dir + npy_filename_head_dens + "%dx%dx%dgrid" % (xdim, ydim, zdim)

            print("\nLoading data from %s" % infile_name)
            dens_gcc, _ = cube_parser.read_cube(infile_name)
            
            print("Saving Density to %s" % (out_filename))
            np.save(out_filename, dens_gcc, allow_pickle=True)
        
        inner_toc = timeit.default_timer()

        print ("\n\nTime %s temp, %s gcc: %4.2f secs" % (temp, gcc, inner_toc - inner_tic))

tot_toc = timeit.default_timer()

print("\n\nSuccess! Total time: %4.2f secs\n\n" % (tot_toc - tot_tic), flush=True)

# FUTURE: Convert to parallel_for over temp and density grid
#if __name__ == '__main__':
#    num_threads = mp.cpu_count()
#    with mp.Pool(processes = N) as p:
#        p.map(read_temp_density_ldos, [infile for infile in ldos_infiles])
