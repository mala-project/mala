#
# An example of accessing Fortran functions via Python ctypes
#

from ctypes import *

n1 = 3
n2 = 4

# C interface

print("")
print("# C interface")

lib_examplec = CDLL("test_libraryc.so",RTLD_GLOBAL)
print("lib_example C",lib_examplec)

lib_examplec.hello_world.restype = None
lib_examplec.hello_world.argtypes = []
lib_examplec.hello_world()

lib_examplec.create_matrix.restype = POINTER(POINTER(c_double))
lib_examplec.create_matrix.argtypes = [c_int, c_int]
lib_examplec.destroy_matrix.restype = None
lib_examplec.destroy_matrix.argtypes = [POINTER(POINTER(c_double))]
array = lib_examplec.create_matrix(n1, n2)
print("array ptr = ", array)
print("array = ")
for i in range(n1):
    for j in range(n2):
        print("%8.4f " % array[i][j],end="")
    print("")
lib_examplec.destroy_matrix(array)

# C++ interface

print("")
print("# C++ interface")

lib_examplecpp = CDLL("test_librarycpp.so",RTLD_GLOBAL)
print("lib_example C++",lib_examplecpp)

lib_examplecpp.hello_world.restype = None
lib_examplecpp.hello_world.argtypes = []
lib_examplecpp.hello_world()

lib_examplecpp.create_matrix.restype = POINTER(POINTER(c_double))
lib_examplecpp.create_matrix.argtypes = [c_int, c_int]
lib_examplecpp.destroy_matrix.restype = None
lib_examplecpp.destroy_matrix.argtypes = [POINTER(POINTER(c_double))]
array = lib_examplecpp.create_matrix(n1, n2)
print("array ptr = ", array)
print("array = ")
for i in range(n1):
    for j in range(n2):
        print("%8.4f " % array[i][j],end="")
    print("")
lib_examplecpp.destroy_matrix(array)

# F90 interface

print("")
print("# F90 interface")

lib_examplef90 = CDLL("test_libraryf90.so",RTLD_GLOBAL)
print("lib_example F90",lib_examplef90)
lib_examplef90.hello_world_.restype = None
lib_examplef90.hello_world_.argtypes = []
lib_examplef90.hello_world_()

lib_examplef90.create_vector_.restype = None
lib_examplef90.create_vector_.argtypes = [c_int, POINTER(c_double)]

# allocate C++ array
array = lib_examplecpp.create_matrix(1, n1)

myvec = array[0]
lib_examplef90.create_vector_(n1, cast(myvec, POINTER(c_double)))
print("myvec ptr = ", myvec)
print("myvec = ")
for i in range(n1):
    print("%8.4f " % myvec[i])

# clean up C++ array
lib_examplecpp.destroy_matrix(array)

lib_examplef90.create_matrix_.restype = None
# create_matrix takes a double*, not a double**, because in Fortran 2D array
# variables are treated like a pointer to a contiguous block of (column-major)
# memory.
lib_examplef90.create_matrix_.argtypes = [c_int, c_int, POINTER(c_double)]

# allocate C++ array
array = lib_examplecpp.create_matrix(n1, n2)
print("array (created by C++) = ")
for i in range(n1):
    for j in range(n2):
        print("%8.4f " % array[i][j],end="")
    print("")

# zero out array
for i in range(n1):
    for j in range(n2):
        array[i][j] = 0.0

# array[0] acts like a double* to the beginning of the block of
# memory
lib_examplef90.create_matrix_(n1, n2, array[0])
print("array (created by Fortran) = ")
for i in range(n1):
    for j in range(n2):
        print("%8.4f " % array[i][j],end="")
    print("")

# clean up C++ array
lib_examplecpp.destroy_matrix(array)

