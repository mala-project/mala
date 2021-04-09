#
# An example of SNAP grid from LAMMPS Python interface
#
# https://lammps.sandia.gov/doc/Python_library.html


from lammps import lammps
import lammps_utils

# define command line input variables

ngridx = 2
ngridy = 4
ngridz = 8
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
try:
    lmp.file("in.bgrid.python")
except lammps.LAMMPSException:
    print("BAD READ")


# get quantities from LAMMPS 

num_atoms = lmp.get_natoms() 

# set things not accessible from LAMMPS

# first 3 cols are x, y, z, coords

ncols0 = 3 

# analytical relation

ncoeff = (twojmax+2)*(twojmax+3)*(twojmax+4)
ncoeff = ncoeff // 24 # integer division
ncols = ncols0+ncoeff
npow = twojmax+1

# power spectrum

powcols = []
count = 0
for j1 in range(0,twojmax+1):
    for j2 in range(0,j1+1):
        for j in range(j1-j2,min(twojmax,j1+j2)+1,2):
            if (j>=j1):
                print(j1,j2,j)
                if j2 == 0:
                    powcols.append(count)
                count += 1

if (count != ncoeff):
    print("ncoeff does not match count")
    exit()
if (len(powcols) != npow):
    print("len(powcols) does not match npow")
    exit()

# Find grid point at atom 2 for this grid
# x: 01 | | 2
# y: 012 | 3 | 4
# z: 01234 | 567 | 8
# 2+2*2+4*8 = 38
iatom2  = 1
igrid_atom2 = 37
igridx_atom2 = 1
igridy_atom2 = 2
igridz_atom2 = 4

# get B_0 at position (0,0,0) in 4 different ways

# 1. from compute sna/atom

bptr = lmp.extract_compute("b", 1, 2) # 1 = per-atom data, 2 = array
print("b = ",bptr[iatom2][0])

# 2. from compute sna/grid

bgridptr = lmp.extract_compute("bgrid", 0, 2) # 0 = style global, 2 = type array
print("bgrid = ",bgridptr[igrid_atom2][ncols0+0])

# 3. from Numpy array pointing to sna/atom array
 
bptr_np = lammps_utils.extract_compute_np(lmp,"b",1,2,(num_atoms,ncoeff))
print("b_np = ",bptr_np[iatom2][0])

# 4. from Numpy array pointing to sna/grid array

bgridptr_np = lammps_utils.extract_compute_np(lmp,"bgrid",0,2,(ngridz,ngridy,ngridx,ncols))
print("bgrid_np = ",bgridptr_np[igridz_atom2][igridy_atom2][igridx_atom2][ncols0+0])

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

# print out the Numpy power spectrum to a file

outfile = open("pgrid_np.dat",'w')
for iz in range(ngridz):
    for iy in range(ngridy):
        for ix in range(ngridx):
            outfile.write("x, y, z = %g %g %g\n" % 
                          (bgridptr_np[iz][iy][ix][0],
                           bgridptr_np[iz][iy][ix][1],
                           bgridptr_np[iz][iy][ix][2]))
            for icoeff in powcols:
                outfile.write("%g " % bgridptr_np[iz][iy][ix][ncols0+icoeff])
            outfile.write("\n")
outfile.close()

# print out the rescaled bispectrum
# use this secret recipe for scaling bispectrum
# This is not guaranteed to be the best choice, but it might help
# 1. Divide all B's by 2j+1
# 2. U_0 = B_(0,0,0)^1/3
# 3. Divide B_(0,0,0) by U_0^3
# 4. Divide B_(j,0,j) by U_0

outfile = open("bgridnorm_np.dat",'w')
for iz in range(ngridz):
    for iy in range(ngridy):
        for ix in range(ngridx):
            outfile.write("x, y, z = %g %g %g\n" % 
                          (bgridptr_np[iz][iy][ix][0],
                           bgridptr_np[iz][iy][ix][1],
                           bgridptr_np[iz][iy][ix][2]))
            icoeff = 0
            val = bgridptr_np[iz][iy][ix][ncols0+icoeff]
            u0inv = val**(-1.0/3.0)
            val = 1.0
            outfile.write("%g " % val)
            icoeff += 1
            for j1 in range(1,twojmax+1):
                for j2 in range(0,j1+1):
                    for j in range(j1-j2,min(twojmax,j1+j2)+1,2):
                        if (j>=j1):
                            fac = 1.0/(j+1)
                            if j2==0: fac *= u0inv
                            val = bgridptr_np[iz][iy][ix][ncols0+icoeff]
                            val *= fac
                            outfile.write("%g " % val)
                            icoeff += 1
            outfile.write("\n")
outfile.close()

outfile = open("pgridnorm_np.dat",'w')
for iz in range(ngridz):
    for iy in range(ngridy):
        for ix in range(ngridx):
            outfile.write("x, y, z = %g %g %g\n" % 
                          (bgridptr_np[iz][iy][ix][0],
                           bgridptr_np[iz][iy][ix][1],
                           bgridptr_np[iz][iy][ix][2]))
            icoeff = 0
            val = bgridptr_np[iz][iy][ix][ncols0+icoeff]
            u0inv = val**(-1.0/3.0)
            val = 1.0
            outfile.write("%g " % val)
            icoeff += 1
            for j1 in range(1,twojmax+1):
                for j2 in range(0,j1+1):
                    for j in range(j1-j2,min(twojmax,j1+j2)+1,2):
                        if (j>=j1):
                            if j2 == 0:
                                fac = 1.0/(j+1)
                                fac *= u0inv
                                val = bgridptr_np[iz][iy][ix][ncols0+icoeff]
                                val *= fac
                                outfile.write("%g " % val)
                            icoeff += 1
            outfile.write("\n")
outfile.close()

