from ctypes import *
import numpy as np

def array2nparray(array, array_shape):
    ptr = array.contents
    total_size = np.prod(array_shape)
    buffer_ptr = cast(ptr, POINTER(c_double * total_size))
    np_array = np.frombuffer(buffer_ptr.contents, dtype=float)
    np_array.shape = array_shape
    return np_array
