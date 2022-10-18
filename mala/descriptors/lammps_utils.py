"""Collection of useful functions for working with LAMMPS."""
import ctypes

import numpy as np


def set_cmdlinevars(cmdargs, argdict):
    """
    Add a dicitionary of LAMMPS arguments in a command line argument string.

    Parameters
    ----------
    cmdargs : list
        Command line argument string. Will be mutated by this function.

    argdict : dict
        Dictionary to be added to LAMMPS command line argument string.

    Returns
    -------
    cmdargs : list
        New command line argument string.
    """
    for key in argdict.keys():
        cmdargs += ["-var", key, f"{argdict[key]}"]
    return cmdargs

# def extract_commands(string):
#     return [x for x in string.splitlines() if x.strip() != '']


def extract_compute_np(lmp, name, compute_type, result_type, array_shape=None):
    """
    Convert a lammps compute to a numpy array.

    Assumes the compute returns floating point numbers.
    Note that the result is a view into the original memory.
    If the result type is 0 (scalar) then conversion to numpy is
    skipped and a python float is returned.

    Parameters
    ----------
    lmp : lammps.lammps
        The LAMMPS object from which data is supposed to be extracted.

    name : string
        Name of the LAMMPS calculation.

    compute_type
        Compute type of the LAMMPS calculation.

    result_type
        Result type of the LAMMPS calculation.

    array_shape
        Array shape of the LAMMPS calculation.

    """
    # 1,2: Style (1) is per-atom compute, returns array type (2).
    ptr = lmp.extract_compute(name, compute_type, result_type)
    if result_type == 0:
        return ptr  # No casting needed, lammps.py already works
    if result_type == 2:
        ptr = ptr.contents
        total_size = np.prod(array_shape)
        buffer_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double *
                                                     total_size))
        array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
        array_np.shape = array_shape
        return array_np
    if result_type == 4 or result_type == 5:
        # ptr is an int
        return ptr
