import os
import sys
import numpy as np
import timeit
import itertools
import argparse

import cube_parser


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--water', action='store_true', default=False,
                            help='run ldos parser on for water files')
args = parser.parse_args()

if (args.water):
    print ("Using this script in water density reading mode")
    folder = '/ascldap/users/jracker/water64cp2k/datast_1593/results/'
    #folder = '/Users/jracker/ldrd_ml_density/mlmm-ldrd-data/water/'

    gcc_grid = [''.join(i) for i in itertools.product("abcdefghijklmnopqrstuvwxyz",repeat=4)][:1593]
    print (gcc_grid)

    cube_filename_head = "w64_"
    cube_filename_tail = "-ELECTRON_DENSITY-1_0.cube"

    # xyz grid
    xdim = 75
    ydim = 75
    zdim = 75

    # only doing density: 1 level
    e_lvls = 1

else:
    folder="/ascldap/users/acangi/q-e_calcs/Al/datasets/RoomTemp/300K/N108/mass-density_highFFTres_e128"

    # Density gram per cubic centimeter grid
    #gcc_grid=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    #gcc_grid={"0.4", "0.6", "0.8", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"}
    #gcc_grid=np.array([0.4, 0.6])
    gcc_grid = {"0.8"}
    # Head and tail of LDOS filenames
#    cube_filename_head = "tmp.pp"
#    cube_filename_tail = "Al_ldos.cube"

    cube_filename_head = "Al_ldos_"
    cube_filename_tail = ".cube"

    # xyz grid
    xdim = 200
    ydim = 200
    zdim = 200

    # Energy levels for LDOS
    e_lvls = 128
    #e_lvls = 2

# Output location of .npy binary files
output_dir = '../ldos_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Output filename head
npy_filename_head = "/ldos_"




temp = 300

xyze_shape = [xdim, ydim, zdim, e_lvls]

tot_tic = timeit.default_timer()
# Loop over density grid
for gcc in gcc_grid:
    out_filename = output_dir + "/%s" + "/%sgcc" % (gcc) + npy_filename_head + "%dx%dx%dgrid_%delvls" % (xdim, ydim, zdim, e_lvls) 
    if (args.water):
        out_filename = folder + cube_filename_head + gcc + cube_filename_tail + "_pkl"

    etic = timeit.default_timer()

    # Allocate space
    ldos_gcc = np.empty(xyze_shape)

    # Loop over energy grid
    # separate cube file for each ldos 
    for e in range(e_lvls):
        infile_name = folder + "/%sgcc/" % (gcc) + \
                      cube_filename_head + "%d" % (e + 1) + cube_filename_tail 
        
        if (args.water):
            infile_name = folder + cube_filename_head + gcc + cube_filename_tail  

        print(infile_name)

        # load into numpy array (ignore metadata w/ '_')
        ldos_gcc[:,:,:,e], _ = cube_parser.read_cube(infile_name)

    
    np.save(out_filename, ldos_gcc, allow_pickle=True)
    
    etoc = timeit.default_timer()

    print ("Time %s gcc: %4.2f secs" % (gcc, etoc - etic))

tot_toc = timeit.default_timer()

print("Success! Total time: %4.2f secs" % (tot_toc - tot_tic))

# FUTURE: Convert to parallel_for over temp and density grid
#if __name__ == '__main__':
#    num_threads = mp.cpu_count()
#    with mp.Pool(processes = N) as p:
#        p.map(read_temp_density_ldos, [infile for infile in ldos_infiles])
