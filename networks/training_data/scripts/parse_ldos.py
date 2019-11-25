import os
import numpy as np
import timeit

import cube_parser

mass_density_folder="/ascldap/users/acangi/q-e_calcs/Al/datasets/RoomTemp/300K/N108/mass-density_highFFTres_e128"

# Density gram per cubic centimeter grid
#gcc_grid=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
gcc_grid=np.array([0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
#gcc_grid=np.array([0.4, 0.6])

# Head and tail of LDOS filenames
cube_filename_head = "tmp.pp"
cube_filename_tail = "Al_ldos.cube"

# xyz grid
xdim = 200
ydim = 200
zdim = 200

# Energy levels for LDOS
e_lvls = 128
#e_lvls = 2

xyze_shape = [xdim, ydim, zdim, e_lvls]

# Output location of .npy binary files
output_dir = '../ldos_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Output filename head
npy_filename_head = "/ldos_"

temp = 300

tot_tic = timeit.default_timer()
# Loop over density grid
for gcc in gcc_grid:
    out_filename = output_dir + npy_filename_head + "%dK_%1.1fgcc_%dx%dx%dgrid_%delvls" % (temp, gcc, xdim, ydim, zdim, e_lvls) 

    etic = timeit.default_timer()

    # Allocate space
    ldos_gcc = np.empty(xyze_shape)

    # Loop over energy grid
    for e in range(e_lvls):
        infile_name = mass_density_folder + "/%1.1fgcc/" % (gcc) + \
                      cube_filename_head + "%03d" % (e + 1) + cube_filename_tail 

        print(infile_name)

        # load into numpy array (ignore metadata w/ '_')
        ldos_gcc[:,:,:,e], _ = cube_parser.read_cube(infile_name)

    
    np.save(out_filename, ldos_gcc, allow_pickle=True)
    
    etoc = timeit.default_timer()

    print ("GCC %1.1f took %4.2f secs" % (gcc, etoc - etic))

tot_toc = timeit.default_timer()

print("Success! Total time: %4.2f secs" % (tot_toc - tot_tic))

# FUTURE: Convert to parallel_for over temp and density grid
#if __name__ == '__main__':
#    num_threads = mp.cpu_count()
#    with mp.Pool(processes = N) as p:
#        p.map(read_temp_density_ldos, [infile for infile in ldos_infiles])
