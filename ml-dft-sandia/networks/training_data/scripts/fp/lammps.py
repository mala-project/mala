# ----------------------------------------------------------------------
#   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
#   http://lammps.sandia.gov, Sandia National Laboratories
#   Steve Plimpton, sjplimp@sandia.gov
#
#   Copyright (2003) Sandia Corporation.  Under the terms of Contract
#   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
#   certain rights in this software.  This software is distributed under
#   the GNU General Public License.
#
#   See the README file in the top-level LAMMPS directory.
# -------------------------------------------------------------------------

# Python wrappers on LAMMPS library via ctypes

# for python3 compatibility

from __future__ import print_function

# imports for simple LAMMPS python wrapper module "lammps"

import sys,traceback,types
from ctypes import *
from os.path import dirname,abspath,join
from inspect import getsourcefile

# imports for advanced LAMMPS python wrapper modules "PyLammps" and "IPyLammps"

from collections import namedtuple
import os
import select
import re
import sys

def get_ctypes_int(size):
  if size == 4:
    return c_int32
  elif size == 8:
    return c_int64
  return c_int

class MPIAbortException(Exception):
  def __init__(self, message):
    self.message = message

  def __str__(self):
    return repr(self.message)

class lammps(object):

  # detect if Python is using version of mpi4py that can pass a communicator

  has_mpi4py = False
  try:
    from mpi4py import MPI
    from mpi4py import __version__ as mpi4py_version
    if mpi4py_version.split('.')[0] in ['2','3']: has_mpi4py = True
  except:
    pass

  # create instance of LAMMPS

  def __init__(self,name="",cmdargs=None,ptr=None,comm=None):
    self.comm = comm
    self.opened = 0

    # determine module location

    modpath = dirname(abspath(getsourcefile(lambda:0)))
    self.lib = None

    # if a pointer to a LAMMPS object is handed in,
    # all symbols should already be available

    try:
      if ptr: self.lib = CDLL("",RTLD_GLOBAL)
    except:
      self.lib = None

    # load liblammps.so unless name is given
    #   if name = "g++", load liblammps_g++.so
    # try loading the LAMMPS shared object from the location
    #   of lammps.py with an absolute path,
    #   so that LD_LIBRARY_PATH does not need to be set for regular install
    # fall back to loading with a relative path,
    #   typically requires LD_LIBRARY_PATH to be set appropriately

    if any([f.startswith('liblammps') and f.endswith('.dylib') for f in os.listdir(modpath)]):
      lib_ext = ".dylib"
    else:
      lib_ext = ".so"

    if not self.lib:
      try:
        if not name: self.lib = CDLL(join(modpath,"liblammps" + lib_ext),RTLD_GLOBAL)
        else: self.lib = CDLL(join(modpath,"liblammps_%s" % name + lib_ext),
                              RTLD_GLOBAL)
      except:
        if not name: self.lib = CDLL("liblammps" + lib_ext,RTLD_GLOBAL)
        else: self.lib = CDLL("liblammps_%s" % name + lib_ext,RTLD_GLOBAL)

    # define ctypes API for each library method
    # NOTE: should add one of these for each lib function

    self.lib.lammps_extract_box.argtypes = \
      [c_void_p,POINTER(c_double),POINTER(c_double),
       POINTER(c_double),POINTER(c_double),POINTER(c_double),
       POINTER(c_int),POINTER(c_int)]
    self.lib.lammps_extract_box.restype = None

    self.lib.lammps_reset_box.argtypes = \
      [c_void_p,POINTER(c_double),POINTER(c_double),c_double,c_double,c_double]
    self.lib.lammps_reset_box.restype = None

    self.lib.lammps_gather_atoms.argtypes = \
      [c_void_p,c_char_p,c_int,c_int,c_void_p]
    self.lib.lammps_gather_atoms.restype = None

    self.lib.lammps_gather_atoms_concat.argtypes = \
      [c_void_p,c_char_p,c_int,c_int,c_void_p]
    self.lib.lammps_gather_atoms_concat.restype = None

    self.lib.lammps_gather_atoms_subset.argtypes = \
      [c_void_p,c_char_p,c_int,c_int,c_int,POINTER(c_int),c_void_p]
    self.lib.lammps_gather_atoms_subset.restype = None

    self.lib.lammps_scatter_atoms.argtypes = \
      [c_void_p,c_char_p,c_int,c_int,c_void_p]
    self.lib.lammps_scatter_atoms.restype = None

    self.lib.lammps_scatter_atoms_subset.argtypes = \
      [c_void_p,c_char_p,c_int,c_int,c_int,POINTER(c_int),c_void_p]
    self.lib.lammps_scatter_atoms_subset.restype = None

    # if no ptr provided, create an instance of LAMMPS
    #   don't know how to pass an MPI communicator from PyPar
    #   but we can pass an MPI communicator from mpi4py v2.0.0 and later
    #   no_mpi call lets LAMMPS use MPI_COMM_WORLD
    #   cargs = array of C strings from args
    # if ptr, then are embedding Python in LAMMPS input script
    #   ptr is the desired instance of LAMMPS
    #   just convert it to ctypes ptr and store in self.lmp

    if not ptr:

      # with mpi4py v2, can pass MPI communicator to LAMMPS
      # need to adjust for type of MPI communicator object
      # allow for int (like MPICH) or void* (like OpenMPI)

      if comm:
        if not lammps.has_mpi4py:
          raise Exception('Python mpi4py version is not 2 or 3')
        if lammps.MPI._sizeof(lammps.MPI.Comm) == sizeof(c_int):
          MPI_Comm = c_int
        else:
          MPI_Comm = c_void_p

        narg = 0
        cargs = 0
        if cmdargs:
          cmdargs.insert(0,"lammps.py")
          narg = len(cmdargs)
          for i in range(narg):
            if type(cmdargs[i]) is str:
              cmdargs[i] = cmdargs[i].encode()
          cargs = (c_char_p*narg)(*cmdargs)
          self.lib.lammps_open.argtypes = [c_int, c_char_p*narg, \
                                           MPI_Comm, c_void_p()]
        else:
          self.lib.lammps_open.argtypes = [c_int, c_int, \
                                           MPI_Comm, c_void_p()]

        self.lib.lammps_open.restype = None
        self.opened = 1
        self.lmp = c_void_p()
        comm_ptr = lammps.MPI._addressof(comm)
        comm_val = MPI_Comm.from_address(comm_ptr)
        self.lib.lammps_open(narg,cargs,comm_val,byref(self.lmp))

      else:
        if lammps.has_mpi4py:
          from mpi4py import MPI
          self.comm = MPI.COMM_WORLD
        self.opened = 1
        if cmdargs:
          cmdargs.insert(0,"lammps.py")
          narg = len(cmdargs)
          for i in range(narg):
            if type(cmdargs[i]) is str:
              cmdargs[i] = cmdargs[i].encode()
          cargs = (c_char_p*narg)(*cmdargs)
          self.lmp = c_void_p()
          self.lib.lammps_open_no_mpi(narg,cargs,byref(self.lmp))
        else:
          self.lmp = c_void_p()
          self.lib.lammps_open_no_mpi(0,None,byref(self.lmp))
          # could use just this if LAMMPS lib interface supported it
          # self.lmp = self.lib.lammps_open_no_mpi(0,None)

    else:
      # magic to convert ptr to ctypes ptr
      if sys.version_info >= (3, 0):
        # Python 3 (uses PyCapsule API)
        pythonapi.PyCapsule_GetPointer.restype = c_void_p
        pythonapi.PyCapsule_GetPointer.argtypes = [py_object, c_char_p]
        self.lmp = c_void_p(pythonapi.PyCapsule_GetPointer(ptr, None))
      else:
        # Python 2 (uses PyCObject API)
        pythonapi.PyCObject_AsVoidPtr.restype = c_void_p
        pythonapi.PyCObject_AsVoidPtr.argtypes = [py_object]
        self.lmp = c_void_p(pythonapi.PyCObject_AsVoidPtr(ptr))

    # optional numpy support (lazy loading)
    self._numpy = None

    # set default types
    self.c_bigint = get_ctypes_int(self.extract_setting("bigint"))
    self.c_tagint = get_ctypes_int(self.extract_setting("tagint"))
    self.c_imageint = get_ctypes_int(self.extract_setting("imageint"))
    self._installed_packages = None

    # add way to insert Python callback for fix external
    self.callback = {}
    self.FIX_EXTERNAL_CALLBACK_FUNC = CFUNCTYPE(None, c_void_p, self.c_bigint, c_int, POINTER(self.c_tagint), POINTER(POINTER(c_double)), POINTER(POINTER(c_double)))
    self.lib.lammps_set_fix_external_callback.argtypes = [c_void_p, c_char_p, self.FIX_EXTERNAL_CALLBACK_FUNC, c_void_p]
    self.lib.lammps_set_fix_external_callback.restype = None

  # shut-down LAMMPS instance

  def __del__(self):
    if self.lmp and self.opened:
      self.lib.lammps_close(self.lmp)
      self.opened = 0

  def close(self):
    if self.opened: self.lib.lammps_close(self.lmp)
    self.lmp = None
    self.opened = 0

  def version(self):
    return self.lib.lammps_version(self.lmp)

  def file(self,file):
    if file: file = file.encode()
    self.lib.lammps_file(self.lmp,file)

  # send a single command

  def command(self,cmd):
    if cmd: cmd = cmd.encode()
    self.lib.lammps_command(self.lmp,cmd)

    if self.has_exceptions and self.lib.lammps_has_error(self.lmp):
      sb = create_string_buffer(100)
      error_type = self.lib.lammps_get_last_error_message(self.lmp, sb, 100)
      error_msg = sb.value.decode().strip()

      if error_type == 2:
        raise MPIAbortException(error_msg)
      raise Exception(error_msg)

  # send a list of commands

  def commands_list(self,cmdlist):
    cmds = [x.encode() for x in cmdlist if type(x) is str]
    args = (c_char_p * len(cmdlist))(*cmds)
    self.lib.lammps_commands_list(self.lmp,len(cmdlist),args)

  # send a string of commands

  def commands_string(self,multicmd):
    if type(multicmd) is str: multicmd = multicmd.encode()
    self.lib.lammps_commands_string(self.lmp,c_char_p(multicmd))

  # extract lammps type byte sizes

  def extract_setting(self, name):
    if name: name = name.encode()
    self.lib.lammps_extract_setting.restype = c_int
    return int(self.lib.lammps_extract_setting(self.lmp,name))

  # extract global info

  def extract_global(self,name,type):
    if name: name = name.encode()
    if type == 0:
      self.lib.lammps_extract_global.restype = POINTER(c_int)
    elif type == 1:
      self.lib.lammps_extract_global.restype = POINTER(c_double)
    else: return None
    ptr = self.lib.lammps_extract_global(self.lmp,name)
    return ptr[0]

  # extract global info

  def extract_box(self):
    boxlo = (3*c_double)()
    boxhi = (3*c_double)()
    xy = c_double()
    yz = c_double()
    xz = c_double()
    periodicity = (3*c_int)()
    box_change = c_int()

    self.lib.lammps_extract_box(self.lmp,boxlo,boxhi,
                                byref(xy),byref(yz),byref(xz),
                                periodicity,byref(box_change))

    boxlo = boxlo[:3]
    boxhi = boxhi[:3]
    xy = xy.value
    yz = yz.value
    xz = xz.value
    periodicity = periodicity[:3]
    box_change = box_change.value

    return boxlo,boxhi,xy,yz,xz,periodicity,box_change

  # extract per-atom info
  # NOTE: need to insure are converting to/from correct Python type
  #   e.g. for Python list or NumPy or ctypes

  def extract_atom(self,name,type):
    if name: name = name.encode()
    if type == 0:
      self.lib.lammps_extract_atom.restype = POINTER(c_int)
    elif type == 1:
      self.lib.lammps_extract_atom.restype = POINTER(POINTER(c_int))
    elif type == 2:
      self.lib.lammps_extract_atom.restype = POINTER(c_double)
    elif type == 3:
      self.lib.lammps_extract_atom.restype = POINTER(POINTER(c_double))
    else: return None
    ptr = self.lib.lammps_extract_atom(self.lmp,name)
    return ptr

  @property
  def numpy(self):
    if not self._numpy:
      import numpy as np
      class LammpsNumpyWrapper:
        def __init__(self, lmp):
          self.lmp = lmp

        def _ctype_to_numpy_int(self, ctype_int):
          if ctype_int == c_int32:
            return np.int32
          elif ctype_int == c_int64:
            return np.int64
          return np.intc

        def extract_atom_iarray(self, name, nelem, dim=1):
          if name in ['id', 'molecule']:
            c_int_type = self.lmp.c_tagint
          elif name in ['image']:
            c_int_type = self.lmp.c_imageint
          else:
            c_int_type = c_int

          np_int_type = self._ctype_to_numpy_int(c_int_type)

          if dim == 1:
            tmp = self.lmp.extract_atom(name, 0)
            ptr = cast(tmp, POINTER(c_int_type * nelem))
          else:
            tmp = self.lmp.extract_atom(name, 1)
            ptr = cast(tmp[0], POINTER(c_int_type * nelem * dim))

          a = np.frombuffer(ptr.contents, dtype=np_int_type)
          a.shape = (nelem, dim)
          return a

        def extract_atom_darray(self, name, nelem, dim=1):
          if dim == 1:
            tmp = self.lmp.extract_atom(name, 2)
            ptr = cast(tmp, POINTER(c_double * nelem))
          else:
            tmp = self.lmp.extract_atom(name, 3)
            ptr = cast(tmp[0], POINTER(c_double * nelem * dim))

          a = np.frombuffer(ptr.contents)
          a.shape = (nelem, dim)
          return a

      self._numpy = LammpsNumpyWrapper(self)
    return self._numpy

  # extract compute info

  def extract_compute(self,id,style,type):
    if id: id = id.encode()
    if type == 0:
      if style > 0: return None
      self.lib.lammps_extract_compute.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_compute(self.lmp,id,style,type)
      return ptr[0]
    if type == 1:
      self.lib.lammps_extract_compute.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_compute(self.lmp,id,style,type)
      return ptr
    if type == 2:
      self.lib.lammps_extract_compute.restype = POINTER(POINTER(c_double))
      ptr = self.lib.lammps_extract_compute(self.lmp,id,style,type)
      return ptr
    return None

  # extract fix info
  # in case of global datum, free memory for 1 double via lammps_free()
  # double was allocated by library interface function

  def extract_fix(self,id,style,type,i=0,j=0):
    if id: id = id.encode()
    if style == 0:
      self.lib.lammps_extract_fix.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_fix(self.lmp,id,style,type,i,j)
      result = ptr[0]
      self.lib.lammps_free(ptr)
      return result
    elif (style == 1) or (style == 2):
      if type == 1:
        self.lib.lammps_extract_fix.restype = POINTER(c_double)
      elif type == 2:
        self.lib.lammps_extract_fix.restype = POINTER(POINTER(c_double))
      else:
        return None
      ptr = self.lib.lammps_extract_fix(self.lmp,id,style,type,i,j)
      return ptr
    else:
      return None

  # extract variable info
  # free memory for 1 double or 1 vector of doubles via lammps_free()
  # for vector, must copy nlocal returned values to local c_double vector
  # memory was allocated by library interface function

  def extract_variable(self,name,group,type):
    if name: name = name.encode()
    if group: group = group.encode()
    if type == 0:
      self.lib.lammps_extract_variable.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_variable(self.lmp,name,group)
      result = ptr[0]
      self.lib.lammps_free(ptr)
      return result
    if type == 1:
      self.lib.lammps_extract_global.restype = POINTER(c_int)
      nlocalptr = self.lib.lammps_extract_global(self.lmp,"nlocal".encode())
      nlocal = nlocalptr[0]
      result = (c_double*nlocal)()
      self.lib.lammps_extract_variable.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_variable(self.lmp,name,group)
      for i in range(nlocal): result[i] = ptr[i]
      self.lib.lammps_free(ptr)
      return result
    return None

  # return current value of thermo keyword

  def get_thermo(self,name):
    if name: name = name.encode()
    self.lib.lammps_get_thermo.restype = c_double
    return self.lib.lammps_get_thermo(self.lmp,name)

  # return total number of atoms in system

  def get_natoms(self):
    return self.lib.lammps_get_natoms(self.lmp)

  # set variable value
  # value is converted to string
  # returns 0 for success, -1 if failed

  def set_variable(self,name,value):
    if name: name = name.encode()
    if value: value = str(value).encode()
    return self.lib.lammps_set_variable(self.lmp,name,value)

  # reset simulation box size

  def reset_box(self,boxlo,boxhi,xy,yz,xz):
    cboxlo = (3*c_double)(*boxlo)
    cboxhi = (3*c_double)(*boxhi)
    self.lib.lammps_reset_box(self.lmp,cboxlo,cboxhi,xy,yz,xz)

  # return vector of atom properties gathered across procs
  # 3 variants to match src/library.cpp
  # name = atom property recognized by LAMMPS in atom->extract()
  # type = 0 for integer values, 1 for double values
  # count = number of per-atom valus, 1 for type or charge, 3 for x or f
  # returned data is a 1d vector - doc how it is ordered?
  # NOTE: need to insure are converting to/from correct Python type
  #   e.g. for Python list or NumPy or ctypes

  def gather_atoms(self,name,type,count):
    if name: name = name.encode()
    natoms = self.lib.lammps_get_natoms(self.lmp)
    if type == 0:
      data = ((count*natoms)*c_int)()
      self.lib.lammps_gather_atoms(self.lmp,name,type,count,data)
    elif type == 1:
      data = ((count*natoms)*c_double)()
      self.lib.lammps_gather_atoms(self.lmp,name,type,count,data)
    else: return None
    return data

  def gather_atoms_concat(self,name,type,count):
    if name: name = name.encode()
    natoms = self.lib.lammps_get_natoms(self.lmp)
    if type == 0:
      data = ((count*natoms)*c_int)()
      self.lib.lammps_gather_atoms_concat(self.lmp,name,type,count,data)
    elif type == 1:
      data = ((count*natoms)*c_double)()
      self.lib.lammps_gather_atoms_concat(self.lmp,name,type,count,data)
    else: return None
    return data

  def gather_atoms_subset(self,name,type,count,ndata,ids):
    if name: name = name.encode()
    if type == 0:
      data = ((count*ndata)*c_int)()
      self.lib.lammps_gather_atoms_subset(self.lmp,name,type,count,ndata,ids,data)
    elif type == 1:
      data = ((count*ndata)*c_double)()
      self.lib.lammps_gather_atoms_subset(self.lmp,name,type,count,ndata,ids,data)
    else: return None
    return data

  # scatter vector of atom properties across procs
  # 2 variants to match src/library.cpp
  # name = atom property recognized by LAMMPS in atom->extract()
  # type = 0 for integer values, 1 for double values
  # count = number of per-atom valus, 1 for type or charge, 3 for x or f
  # assume data is of correct type and length, as created by gather_atoms()
  # NOTE: need to insure are converting to/from correct Python type
  #   e.g. for Python list or NumPy or ctypes

  def scatter_atoms(self,name,type,count,data):
    if name: name = name.encode()
    self.lib.lammps_scatter_atoms(self.lmp,name,type,count,data)

  def scatter_atoms_subset(self,name,type,count,ndata,ids,data):
    if name: name = name.encode()
    self.lib.lammps_scatter_atoms_subset(self.lmp,name,type,count,ndata,ids,data)

  # create N atoms on all procs
  # N = global number of atoms
  # id = ID of each atom (optional, can be None)
  # type = type of each atom (1 to Ntypes) (required)
  # x = coords of each atom as (N,3) array (required)
  # v = velocity of each atom as (N,3) array (optional, can be None)
  # NOTE: how could we insure are passing correct type to LAMMPS
  #   e.g. for Python list or NumPy, etc
  #   ditto for gather_atoms() above

  def create_atoms(self,n,id,type,x,v,image=None,shrinkexceed=False):
    if id:
      id_lmp = (c_int * n)()
      id_lmp[:] = id
    else:
      id_lmp = id

    if image:
      image_lmp = (c_int * n)()
      image_lmp[:] = image
    else:
      image_lmp = image

    type_lmp = (c_int * n)()
    type_lmp[:] = type
    self.lib.lammps_create_atoms(self.lmp,n,id_lmp,type_lmp,x,v,image_lmp,
                                 shrinkexceed)

  @property
  def has_exceptions(self):
    """ Return whether the LAMMPS shared library was compiled with C++ exceptions handling enabled """
    return self.lib.lammps_config_has_exceptions() != 0

  @property
  def has_gzip_support(self):
    return self.lib.lammps_config_has_gzip_support() != 0

  @property
  def has_png_support(self):
    return self.lib.lammps_config_has_png_support() != 0

  @property
  def has_jpeg_support(self):
    return self.lib.lammps_config_has_jpeg_support() != 0

  @property
  def has_ffmpeg_support(self):
    return self.lib.lammps_config_has_ffmpeg_support() != 0

  @property
  def installed_packages(self):
    if self._installed_packages is None:
      self._installed_packages = []
      npackages = self.lib.lammps_config_package_count()
      sb = create_string_buffer(100)
      for idx in range(npackages):
        self.lib.lammps_config_package_name(idx, sb, 100)
        self._installed_packages.append(sb.value.decode())
    return self._installed_packages

  def set_fix_external_callback(self, fix_name, callback, caller=None):
    import numpy as np
    def _ctype_to_numpy_int(ctype_int):
          if ctype_int == c_int32:
            return np.int32
          elif ctype_int == c_int64:
            return np.int64
          return np.intc

    def callback_wrapper(caller_ptr, ntimestep, nlocal, tag_ptr, x_ptr, fext_ptr):
      if cast(caller_ptr,POINTER(py_object)).contents:
        pyCallerObj = cast(caller_ptr,POINTER(py_object)).contents.value
      else:
        pyCallerObj = None

      tptr = cast(tag_ptr, POINTER(self.c_tagint * nlocal))
      tag = np.frombuffer(tptr.contents, dtype=_ctype_to_numpy_int(self.c_tagint))
      tag.shape = (nlocal)

      xptr = cast(x_ptr[0], POINTER(c_double * nlocal * 3))
      x = np.frombuffer(xptr.contents)
      x.shape = (nlocal, 3)

      fptr = cast(fext_ptr[0], POINTER(c_double * nlocal * 3))
      f = np.frombuffer(fptr.contents)
      f.shape = (nlocal, 3)

      callback(pyCallerObj, ntimestep, nlocal, tag, x, f)

    cFunc   = self.FIX_EXTERNAL_CALLBACK_FUNC(callback_wrapper)
    cCaller = cast(pointer(py_object(caller)), c_void_p)

    self.callback[fix_name] = { 'function': cFunc, 'caller': caller }

    self.lib.lammps_set_fix_external_callback(self.lmp, fix_name.encode(), cFunc, cCaller)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

################################################################################
# Alternative Python Wrapper
# Written by Richard Berger <richard.berger@temple.edu>
################################################################################

class OutputCapture(object):
  """ Utility class to capture LAMMPS library output """

  def __init__(self):
    self.stdout_pipe_read, self.stdout_pipe_write = os.pipe()
    self.stdout_fd = 1

  def __enter__(self):
    self.stdout = os.dup(self.stdout_fd)
    os.dup2(self.stdout_pipe_write, self.stdout_fd)
    return self

  def __exit__(self, type, value, tracebac):
    os.dup2(self.stdout, self.stdout_fd)
    os.close(self.stdout)
    os.close(self.stdout_pipe_read)
    os.close(self.stdout_pipe_write)

  # check if we have more to read from the pipe
  def more_data(self, pipe):
    r, _, _ = select.select([pipe], [], [], 0)
    return bool(r)

  # read the whole pipe
  def read_pipe(self, pipe):
    out = ""
    while self.more_data(pipe):
      out += os.read(pipe, 1024).decode()
    return out

  @property
  def output(self):
    return self.read_pipe(self.stdout_pipe_read)


class Variable(object):
  def __init__(self, lammps_wrapper_instance, name, style, definition):
    self.wrapper = lammps_wrapper_instance
    self.name = name
    self.style = style
    self.definition = definition.split()

  @property
  def value(self):
    if self.style == 'atom':
      return list(self.wrapper.lmp.extract_variable(self.name, "all", 1))
    else:
      value = self.wrapper.lmp_print('"${%s}"' % self.name).strip()
      try:
        return float(value)
      except ValueError:
        return value


class AtomList(object):
  def __init__(self, lammps_wrapper_instance):
    self.lmp = lammps_wrapper_instance
    self.natoms = self.lmp.system.natoms
    self.dimensions = self.lmp.system.dimensions

  def __getitem__(self, index):
    if self.dimensions == 2:
        return Atom2D(self.lmp, index + 1)
    return Atom(self.lmp, index + 1)


class Atom(object):
  def __init__(self, lammps_wrapper_instance, index):
    self.lmp = lammps_wrapper_instance
    self.index = index

  @property
  def id(self):
    return int(self.lmp.eval("id[%d]" % self.index))

  @property
  def type(self):
    return int(self.lmp.eval("type[%d]" % self.index))

  @property
  def mol(self):
    return self.lmp.eval("mol[%d]" % self.index)

  @property
  def mass(self):
    return self.lmp.eval("mass[%d]" % self.index)

  @property
  def position(self):
    return (self.lmp.eval("x[%d]" % self.index),
            self.lmp.eval("y[%d]" % self.index),
            self.lmp.eval("z[%d]" % self.index))

  @position.setter
  def position(self, value):
     self.lmp.set("atom", self.index, "x", value[0])
     self.lmp.set("atom", self.index, "y", value[1])
     self.lmp.set("atom", self.index, "z", value[2])

  @property
  def velocity(self):
    return (self.lmp.eval("vx[%d]" % self.index),
            self.lmp.eval("vy[%d]" % self.index),
            self.lmp.eval("vz[%d]" % self.index))

  @velocity.setter
  def velocity(self, value):
     self.lmp.set("atom", self.index, "vx", value[0])
     self.lmp.set("atom", self.index, "vy", value[1])
     self.lmp.set("atom", self.index, "vz", value[2])

  @property
  def force(self):
    return (self.lmp.eval("fx[%d]" % self.index),
            self.lmp.eval("fy[%d]" % self.index),
            self.lmp.eval("fz[%d]" % self.index))

  @property
  def charge(self):
    return self.lmp.eval("q[%d]" % self.index)


class Atom2D(Atom):
  def __init__(self, lammps_wrapper_instance, index):
    super(Atom2D, self).__init__(lammps_wrapper_instance, index)

  @property
  def position(self):
    return (self.lmp.eval("x[%d]" % self.index),
            self.lmp.eval("y[%d]" % self.index))

  @position.setter
  def position(self, value):
     self.lmp.set("atom", self.index, "x", value[0])
     self.lmp.set("atom", self.index, "y", value[1])

  @property
  def velocity(self):
    return (self.lmp.eval("vx[%d]" % self.index),
            self.lmp.eval("vy[%d]" % self.index))

  @velocity.setter
  def velocity(self, value):
     self.lmp.set("atom", self.index, "vx", value[0])
     self.lmp.set("atom", self.index, "vy", value[1])

  @property
  def force(self):
    return (self.lmp.eval("fx[%d]" % self.index),
            self.lmp.eval("fy[%d]" % self.index))


class variable_set:
    def __init__(self, name, variable_dict):
        self._name = name
        array_pattern = re.compile(r"(?P<arr>.+)\[(?P<index>[0-9]+)\]")

        for key, value in variable_dict.items():
            m = array_pattern.match(key)
            if m:
                g = m.groupdict()
                varname = g['arr']
                idx = int(g['index'])
                if varname not in self.__dict__:
                    self.__dict__[varname] = {}
                self.__dict__[varname][idx] = value
            else:
                self.__dict__[key] = value

    def __str__(self):
        return "{}({})".format(self._name, ','.join(["{}={}".format(k, self.__dict__[k]) for k in self.__dict__.keys() if not k.startswith('_')]))

    def __repr__(self):
        return self.__str__()


def get_thermo_data(output):
    """ traverse output of runs and extract thermo data columns """
    if isinstance(output, str):
        lines = output.splitlines()
    else:
        lines = output

    runs = []
    columns = []
    in_run = False
    current_run = {}

    for line in lines:
        if line.startswith("Per MPI rank memory allocation"):
            in_run = True
        elif in_run and len(columns) == 0:
            # first line after memory usage are column names
            columns = line.split()

            current_run = {}

            for col in columns:
                current_run[col] = []

        elif line.startswith("Loop time of "):
            in_run = False
            columns = None
            thermo_data = variable_set('ThermoData', current_run)
            r = {'thermo' : thermo_data }
            runs.append(namedtuple('Run', list(r.keys()))(*list(r.values())))
        elif in_run and len(columns) > 0:
            values = [float(x) for x in line.split()]

            for i, col in enumerate(columns):
                current_run[col].append(values[i])
    return runs

class PyLammps(object):
  """
  More Python-like wrapper for LAMMPS (e.g., for iPython)
  See examples/ipython for usage
  """

  def __init__(self,name="",cmdargs=None,ptr=None,comm=None):
    if ptr:
      if isinstance(ptr,PyLammps):
        self.lmp = ptr.lmp
      elif isinstance(ptr,lammps):
        self.lmp = ptr
      else:
        self.lmp = lammps(name=name,cmdargs=cmdargs,ptr=ptr,comm=comm)
    else:
      self.lmp = lammps(name=name,cmdargs=cmdargs,ptr=None,comm=comm)
    print("LAMMPS output is captured by PyLammps wrapper")
    self._cmd_history = []
    self.runs = []

  def __del__(self):
    if self.lmp: self.lmp.close()
    self.lmp = None

  def close(self):
    if self.lmp: self.lmp.close()
    self.lmp = None

  def version(self):
    return self.lmp.version()

  def file(self,file):
    self.lmp.file(file)

  def write_script(self,filename):
    """ Write LAMMPS script file containing all commands executed up until now """
    with open(filename, "w") as f:
      for cmd in self._cmd_history:
        f.write("%s\n" % cmd)

  def command(self,cmd):
    self.lmp.command(cmd)
    self._cmd_history.append(cmd)

  def run(self, *args, **kwargs):
    output = self.__getattr__('run')(*args, **kwargs)

    if(lammps.has_mpi4py):
      output = self.lmp.comm.bcast(output, root=0)

    self.runs += get_thermo_data(output)
    return output

  @property
  def last_run(self):
    if len(self.runs) > 0:
        return self.runs[-1]
    return None

  @property
  def atoms(self):
    return AtomList(self)

  @property
  def system(self):
    output = self.info("system")
    d = self._parse_info_system(output)
    return namedtuple('System', d.keys())(*d.values())

  @property
  def communication(self):
    output = self.info("communication")
    d = self._parse_info_communication(output)
    return namedtuple('Communication', d.keys())(*d.values())

  @property
  def computes(self):
    output = self.info("computes")
    return self._parse_element_list(output)

  @property
  def dumps(self):
    output = self.info("dumps")
    return self._parse_element_list(output)

  @property
  def fixes(self):
    output = self.info("fixes")
    return self._parse_element_list(output)

  @property
  def groups(self):
    output = self.info("groups")
    return self._parse_groups(output)

  @property
  def variables(self):
    output = self.info("variables")
    vars = {}
    for v in self._parse_element_list(output):
      vars[v['name']] = Variable(self, v['name'], v['style'], v['def'])
    return vars

  def eval(self, expr):
    value = self.lmp_print('"$(%s)"' % expr).strip()
    try:
      return float(value)
    except ValueError:
      return value

  def _split_values(self, line):
    return [x.strip() for x in line.split(',')]

  def _get_pair(self, value):
    return [x.strip() for x in value.split('=')]

  def _parse_info_system(self, output):
    lines = output[6:-2]
    system = {}

    for line in lines:
      if line.startswith("Units"):
        system['units'] = self._get_pair(line)[1]
      elif line.startswith("Atom style"):
        system['atom_style'] = self._get_pair(line)[1]
      elif line.startswith("Atom map"):
        system['atom_map'] = self._get_pair(line)[1]
      elif line.startswith("Atoms"):
        parts = self._split_values(line)
        system['natoms'] = int(self._get_pair(parts[0])[1])
        system['ntypes'] = int(self._get_pair(parts[1])[1])
        system['style'] = self._get_pair(parts[2])[1]
      elif line.startswith("Kspace style"):
        system['kspace_style'] = self._get_pair(line)[1]
      elif line.startswith("Dimensions"):
        system['dimensions'] = int(self._get_pair(line)[1])
      elif line.startswith("Orthogonal box"):
        system['orthogonal_box'] = [float(x) for x in self._get_pair(line)[1].split('x')]
      elif line.startswith("Boundaries"):
        system['boundaries'] = self._get_pair(line)[1]
      elif line.startswith("xlo"):
        keys, values = [self._split_values(x) for x in self._get_pair(line)]
        for key, value in zip(keys, values):
          system[key] = float(value)
      elif line.startswith("ylo"):
        keys, values = [self._split_values(x) for x in self._get_pair(line)]
        for key, value in zip(keys, values):
          system[key] = float(value)
      elif line.startswith("zlo"):
        keys, values = [self._split_values(x) for x in self._get_pair(line)]
        for key, value in zip(keys, values):
          system[key] = float(value)
      elif line.startswith("Molecule type"):
        system['molecule_type'] = self._get_pair(line)[1]
      elif line.startswith("Bonds"):
        parts = self._split_values(line)
        system['nbonds'] = int(self._get_pair(parts[0])[1])
        system['nbondtypes'] = int(self._get_pair(parts[1])[1])
        system['bond_style'] = self._get_pair(parts[2])[1]
      elif line.startswith("Angles"):
        parts = self._split_values(line)
        system['nangles'] = int(self._get_pair(parts[0])[1])
        system['nangletypes'] = int(self._get_pair(parts[1])[1])
        system['angle_style'] = self._get_pair(parts[2])[1]
      elif line.startswith("Dihedrals"):
        parts = self._split_values(line)
        system['ndihedrals'] = int(self._get_pair(parts[0])[1])
        system['ndihedraltypes'] = int(self._get_pair(parts[1])[1])
        system['dihedral_style'] = self._get_pair(parts[2])[1]
      elif line.startswith("Impropers"):
        parts = self._split_values(line)
        system['nimpropers'] = int(self._get_pair(parts[0])[1])
        system['nimpropertypes'] = int(self._get_pair(parts[1])[1])
        system['improper_style'] = self._get_pair(parts[2])[1]

    return system

  def _parse_info_communication(self, output):
    lines = output[6:-3]
    comm = {}

    for line in lines:
      if line.startswith("MPI library"):
        comm['mpi_version'] = line.split(':')[1].strip()
      elif line.startswith("Comm style"):
        parts = self._split_values(line)
        comm['comm_style'] = self._get_pair(parts[0])[1]
        comm['comm_layout'] = self._get_pair(parts[1])[1]
      elif line.startswith("Processor grid"):
        comm['proc_grid'] = [int(x) for x in self._get_pair(line)[1].split('x')]
      elif line.startswith("Communicate velocities for ghost atoms"):
        comm['ghost_velocity'] = (self._get_pair(line)[1] == "yes")
      elif line.startswith("Nprocs"):
        parts = self._split_values(line)
        comm['nprocs'] = int(self._get_pair(parts[0])[1])
        comm['nthreads'] = int(self._get_pair(parts[1])[1])
    return comm

  def _parse_element_list(self, output):
    lines = output[6:-3]
    elements = []

    for line in lines:
      element_info = self._split_values(line.split(':')[1].strip())
      element = {'name': element_info[0]}
      for key, value in [self._get_pair(x) for x in element_info[1:]]:
        element[key] = value
      elements.append(element)
    return elements

  def _parse_groups(self, output):
    lines = output[6:-3]
    groups = []
    group_pattern = re.compile(r"(?P<name>.+) \((?P<type>.+)\)")

    for line in lines:
      m = group_pattern.match(line.split(':')[1].strip())
      group = {'name': m.group('name'), 'type': m.group('type')}
      groups.append(group)
    return groups

  def lmp_print(self, s):
    """ needed for Python2 compatibility, since print is a reserved keyword """
    return self.__getattr__("print")(s)

  def __dir__(self):
    return ['angle_coeff', 'angle_style', 'atom_modify', 'atom_style', 'atom_style',
    'bond_coeff', 'bond_style', 'boundary', 'change_box', 'communicate', 'compute',
    'create_atoms', 'create_box', 'delete_atoms', 'delete_bonds', 'dielectric',
    'dihedral_coeff', 'dihedral_style', 'dimension', 'dump', 'fix', 'fix_modify',
    'group', 'improper_coeff', 'improper_style', 'include', 'kspace_modify',
    'kspace_style', 'lattice', 'mass', 'minimize', 'min_style', 'neighbor',
    'neigh_modify', 'newton', 'nthreads', 'pair_coeff', 'pair_modify',
    'pair_style', 'processors', 'read', 'read_data', 'read_restart', 'region',
    'replicate', 'reset_timestep', 'restart', 'run', 'run_style', 'thermo',
    'thermo_modify', 'thermo_style', 'timestep', 'undump', 'unfix', 'units',
    'variable', 'velocity', 'write_restart']

  def __getattr__(self, name):
    def handler(*args, **kwargs):
      cmd_args = [name] + [str(x) for x in args]

      with OutputCapture() as capture:
        self.command(' '.join(cmd_args))
        output = capture.output

      if 'verbose' in kwargs and kwargs['verbose']:
        print(output)

      lines = output.splitlines()

      if len(lines) > 1:
        return lines
      elif len(lines) == 1:
        return lines[0]
      return None

    return handler


class IPyLammps(PyLammps):
  """
  iPython wrapper for LAMMPS which adds embedded graphics capabilities
  """

  def __init__(self,name="",cmdargs=None,ptr=None,comm=None):
    super(IPyLammps, self).__init__(name=name,cmdargs=cmdargs,ptr=ptr,comm=comm)

  def image(self, filename="snapshot.png", group="all", color="type", diameter="type",
            size=None, view=None, center=None, up=None, zoom=1.0):
    cmd_args = [group, "image", filename, color, diameter]

    if size:
      width = size[0]
      height = size[1]
      cmd_args += ["size", width, height]

    if view:
      theta = view[0]
      phi = view[1]
      cmd_args += ["view", theta, phi]

    if center:
      flag = center[0]
      Cx = center[1]
      Cy = center[2]
      Cz = center[3]
      cmd_args += ["center", flag, Cx, Cy, Cz]

    if up:
      Ux = up[0]
      Uy = up[1]
      Uz = up[2]
      cmd_args += ["up", Ux, Uy, Uz]

    if zoom:
      cmd_args += ["zoom", zoom]

    cmd_args.append("modify backcolor white")

    self.write_dump(*cmd_args)
    from IPython.core.display import Image
    return Image('snapshot.png')

  def video(self, filename):
    from IPython.display import HTML
    return HTML("<video controls><source src=\"" + filename + "\"></video>")
