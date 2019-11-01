#
# An example of accessing Fortran functions via Python ctypes
#

from ctypes import *

# C interface

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
n1 = 2
n2 = 3
array = lib_examplec.create_matrix(n1, n2)
print("array ptr = ", array)
print("array = ")
for i in range(n1):
    for j in range(n2):
        print("%g " % array[i][j])
    print("")
lib_examplec.destroy_matrix(array)

# C++ interface

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
n1 = 2
n2 = 3
array = lib_examplecpp.create_matrix(n1, n2)
print("array ptr = ", array)
print("array = ")
for i in range(n1):
    for j in range(n2):
        print("%g " % array[i][j])
    print("")
lib_examplecpp.destroy_matrix(array)

# F90 interface

print("# F90 interface")

lib_examplef90 = CDLL("test_libraryf90.so",RTLD_GLOBAL)
print("lib_example F90",lib_examplef90)
lib_examplef90.hello_world_.restype = None
lib_examplef90.hello_world_.argtypes = []
lib_examplef90.hello_world_()

lib_examplef90.create_vector_.restype = None
lib_examplef90.create_vector_.argtypes = [c_int, POINTER(c_double)]
n = 3

# allocate C++ array
array = lib_examplecpp.create_matrix(1, n)

myvec = array[0]
lib_examplef90.create_vector_(n, cast(myvec, POINTER(c_double)))
print("myvec ptr = ", myvec)
print("myvec = ")
for i in range(n):
    print("%g " % myvec[i])

# clean up C++ array
lib_examplecpp.destroy_matrix(array)

lib_examplef90.create_matrix_.restype = None
lib_examplef90.create_matrix_.argtypes = [c_int, c_int, POINTER(POINTER(c_double))]
n1 = 3
n2 = 3

# allocate C++ array
array = lib_examplecpp.create_matrix(n1, n2)
print("array ptr = ", array,array[0],array[1],array[2],array[0][0])

# This does not work, gies segfault

#lib_examplef90.create_matrix_(n1, n2, array)
#print("array ptr = ", array,array[0],array[1],array[2])
#print("array = ")
#for i in range(n1):
#    for j in range(n2):
#        print("%g " % array[i][j])
#    print("")

# clean up C++ array
lib_examplecpp.destroy_matrix(array)

