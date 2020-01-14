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
parser.add_argument('--water', action='store_true', default=False,
                            help='run ldos parser on for water files')
parser.add_argument('--run-all', action='store_true', default=False,
                            help='run ldos parser for all densities and temperatures')
parser.add_argument('--density', action='store_true', default=False,
                            help='run density parser for each density and temperature, also')
parser.add_argument('--nxyz', type=int, default=200, metavar='N',
                            help='number of grid cells in the X/Y/Z dimensions (default: 200)')
parser.add_argument('--temp', type=str, default="300K", metavar='T',
                            help='temperature of ldos parser in units K (default: 300K)')
parser.add_argument('--gcc', type=str, default="2.0", metavar='GCC',
                            help='density of ldos parser in units g/cm^3 (default: "2.0")')
parser.add_argument('--elvls', type=int, default=1, metavar='E',
                            help='the number of energy levels in the LDOS (for density only e=1) (default: 1)')
parser.add_argument('--data-dir', type=str, \
        default="/ascldap/users/acangi/q-e_calcs/Al/datasets/RoomTemp/300K/N108/mass-density_highFFTres_e128", \
        metavar="str", help='path to data directory (default: Attila Al N108 directory)')
parser.add_argument('--output-dir', type=str, default="../ldos_data",
        metavar="str", help='path to output directory (default: ../ldos_data)')
args = parser.parse_args()

if (args.water):
    print ("Using this script in water density reading mode")
    args.data_dir = '/ascldap/users/jracker/water64cp2k/datast_1593/results/'
    #args.data_dir = '/Users/jracker/ldrd_ml_density/mlmm-ldrd-data/water/'

    temp_grid = np.array([args.temp])
    gcc_grid = np.array(['aaaa'])

    if (args.run_all):
        gcc_grid = [''.join(i) for i in itertools.product("abcdefghijklmnopqrstuvwxyz",repeat=4)][:1593]
    
    print (gcc_grid)

    cube_filename_head = "w64_"
    cube_filename_tail = "-ELECTRON_DENSITY-1_0.cube"

    # xyz grid
    xdim = args.nxyz
    ydim = xdim
    zdim = ydim

    # only doing density: 1 level
    e_lvls = args.elvls

else:
    # Density gram per cubic centimeter grid
   
    if (not args.run_all):
        temp_grid = np.array([args.temp])
        gcc_grid = np.array([args.gcc])
    else: 
        
        # Currently available
        temp_grid = np.array(["300K"])
        # Future
#        temp_grid = np.array(["300K", "10000K", "20000K", "30000K"])
        
        # Currently available
        gcc_grid = np.array(["0.4", "0.6", "0.8", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"])
        # Future
#        gcc_grid = np.array(["0.1", "0.2", "0.4", "0.6", "0.8", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"])
        # Test
#        gcc_grid = np.array(["0.4", "0.6"])

    # Head and tail of LDOS filenames
    cube_filename_head = "Al_ldos_"
    cube_filename_tail = ".cube"

    dens_filename = "Al_dens.cube"

    # xyz grid
    xdim = args.nxyz
    ydim = xdim
    zdim = ydim

    # Energy levels for LDOS
    e_lvls = args.elvls

# Output location of .npy binary files
#output_dir = '../ldos_data'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Output filename head
npy_filename_head = "/ldos_"
npy_filename_head_dens = "/dens_"

xyze_shape = [xdim, ydim, zdim, e_lvls]
dens_shape = [xdim, ydim, zdim]

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

        out_filename = args.gcc_dir + npy_filename_head + "%dx%dx%dgrid_%delvls" % (xdim, ydim, zdim, e_lvls) 

        if (args.water):
            out_filename = args.data_dir + cube_filename_head + gcc + cube_filename_tail + "_pkl"
        else:
            # Make Temp directory
            if not os.path.exists(temp_dir):
                print("\nCreating folder %s" % temp_dir)
                os.makedirs(temp_dir)
            # Make Density directory
            if not os.path.exists(gcc_dir):
                print("\nCreating folder %s" % gcc_dir)
                os.makedirs(gcc_dir)

        # Allocate space
        ldos_gcc = np.empty(xyze_shape)

        # Loop over energy grid
        # separate cube file for each ldos 
        for e in range(e_lvls):
            infile_name = args.data_dir + "/%sgcc/" % (gcc) + \
                          cube_filename_head + "%d" % (e) + cube_filename_tail

            if (args.water):
                infile_name = args.data_dir + cube_filename_head + gcc + cube_filename_tail  

            print("\nLoading data from %s" % infile_name)

            # load into numpy array (ignore metadata w/ '_')
            ldos_gcc[:,:,:,e], _ = cube_parser.read_cube(infile_name)

        # Save numpy array to binary file

        print("Saving LDOS to %s" % (out_filename))
        np.save(out_filename, ldos_gcc, allow_pickle=True)
       
        # If parsing density, as well
        if (args.density):
            infile_name = args.data_dir + "/%sgcc/" % (gcc) + dens_filename
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
