#
# An example of SNAP grid from LAMMPS Python interface
#
# https://lammps.sandia.gov/doc/Python_library.html


from lammps import lammps
import lammps_utils

import ctypes
#so_file = "helloworld.o"
# so_file = "helloworld.so"
#lib_example = ctypes.cdll.LoadLibrary(so_file)
#lib_example = CDLL("helloworldc.so",RTLD_GLOBAL)
#lib_example = CDLL("helloworldc.so",RTLD_GLOBAL)
#lib_example = ctypes.CDLL("helloworldc.so")
lib_example = ctypes.CDLL("helloworldf90.so")
print("lib_example ",lib_example)

lib_example.hello_world_.restype = None
lib_example.hello_world_.argtypes = []
print("lib_example.hello_world()",lib_example.hello_world_)
lib_example.hello_world_()

exit()

# define command line input variables

ngridx = 2
ngridy = 3
ngridz = 4
twojmax = 2

lmp_cmdargs = ["-echo","screen"]
lmp_cmdargs = lammps_utils.set_cmdlinevars(lmp_cmdargs,
    {
        "ngridx":ngridx,
        "ngridy":ngridy,
        "ngridz":ngridz,
        "twojmax":twojmax
        }
    )

# launch LAMMPS instance

lmp = lammps(cmdargs=lmp_cmdargs)

# run LAMMPS input script

lmp.file("in.bgrid.python")

# get quantities from LAMMPS 

num_atoms = lmp.get_natoms() 

# set things not accessible from LAMMPS

# first 3 cols are x, y, z, coords

ncols0 = 3 

# analytical relation

ncoeff = (twojmax+2)*(twojmax+3)*(twojmax+4)
ncoeff = ncoeff // 24 # integer division
ncols = ncols0+ncoeff

# get B_0 at position (0,0,0) in 4 different ways

# 1. from comute sna/atom

bptr = lmp.extract_compute("b", 1, 2) # 1 = per-atom data, 2 = array
print("b = ",bptr[0][0])

# 2. from compute sna/grid

bgridptr = lmp.extract_compute("bgrid", 0, 2) # 0 = style global, 2 = type array
print("bgrid = ",bgridptr[0][ncols0+0])

# 3. from Numpy array pointing to sna/atom array
 
bptr_np = lammps_utils.extract_compute_np(lmp,"b",1,2,(num_atoms,ncoeff))
print("b_np = ",bptr_np[0][0])

# 4. from Numpy array pointing to sna/grid array

bgridptr_np = lammps_utils.extract_compute_np(lmp,"bgrid",0,2,(ngridz,ngridy,ngridx,ncols))
print("bgrid_np = ",bgridptr_np[0][0][0][ncols0+0])

# print out the LAMMPS array to a file

outfile = open("bgrid.dat",'w')
igrid = 0
for iz in range(ngridz):
    for iy in range(ngridy):
        for ix in range(ngridx):
            outfile.write("x, y, z = %g %g %g\n" % 
                          (bgridptr[igrid][0],
                           bgridptr[igrid][1],
                           bgridptr[igrid][2]))
            for icoeff in range(ncoeff):
                outfile.write("%g " % bgridptr[igrid][ncols0+icoeff])
            outfile.write("\n")
            igrid += 1
outfile.close()

# print out the Numpy array to a file

outfile = open("bgrid_np.dat",'w')
for iz in range(ngridz):
    for iy in range(ngridy):
        for ix in range(ngridx):
            outfile.write("x, y, z = %g %g %g\n" % 
                          (bgridptr_np[iz][iy][ix][0],
                           bgridptr_np[iz][iy][ix][1],
                           bgridptr_np[iz][iy][ix][2]))
            for icoeff in range(ncoeff):
                outfile.write("%g " % bgridptr_np[iz][iy][ix][ncols0+icoeff])
            outfile.write("\n")
outfile.close()
