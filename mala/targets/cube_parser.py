"""
Cube parser, taken from cubetools (see below).

------------------------------------------------------------------------------

Module: cubetools

------------------------------------------------------------------------------

Description:
Module to work with Gaussian cube format files
(see http://paulbourke.net/dataformats/cube/)

------------------------------------------------------------------------------

What does it do:

- Read/write cube files to/from numpy arrays (dtype=float*)
- Read/write pairse of cube files to/from numpy arrays (dtype=complex*)
- Provides a CubeFile object, to be used when cubefiles with constant and
  static data is required. It simulates the readline method
  of a file object with a cube file opened, without creating a file

------------------------------------------------------------------------------

Dependency: numpy

------------------------------------------------------------------------------

Author: P. R. Vaidyanathan (aditya95sriram <at> gmail <dot> com)
Date: 25th June 2017

------------------------------------------------------------------------------

MIT License

Copyright (c) 2019 P. R. Vaidyanathan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------
"""
import numpy as np

if __name__ == '__main__':
    DEBUGMODE = True
else:
    DEBUGMODE = False


def _debug(*args):
    global DEBUGMODE
#    if DEBUGMODE:
#        print " ".join(map(str, args))


class CubeFile(object):
    """
    Object which mimics a cube file opened as a file object.

    Done by returning output in the correct format, matching the
    metadata of the source cube file and replacing volumetric
    data with static data provided as arg to the constructor. 
    Doesn't copy atoms metadata, retains number of atoms, but
    returns dummy atoms
    Mimics file object's readline method.

    Parameters
    ----------
    srcname: string
        source file to copy metadata from

    const: int
        numeric value to return instead of volumetric data
    """

    def __init__(self, srcname, const=1):
        self.cursor = 0
        self.const = const
        self.src = src = open(srcname)
        # comments
        src.readline()
        src.readline()
        _debug(srcname)
        self.lines = [" Cubefile created by cubetools.py\n", 
                      "  source: {0}\n".format(srcname)]
        self.lines.append(src.readline())  # read natm and origin
        self.natm = int(self.lines[-1].strip().split()[0])
        # read cube dim and vectors along 3 axes
        self.lines.extend(src.readline() for i in range(3))
        self.src.close()
        self.nx, self.ny, self.nz = [int(line.strip().split()[0])
                                     for line in self.lines[3:6]]
        self.remvals = self.nz
        self.remrows = self.nx*self.ny
        for i in range(self.natm):
            self.lines.append("{0:^ 8d}".format(1) + "{0:< 12.6f}".format(0)*4
                              + '\n')

    def __del__(self):
        """Close Cube file."""
        self.src.close()

    def readline(self):
        """
        Read next line.

        Mimic readline method of file object with cube file opened.

        Returns
        -------
        retval : string
            Current line.
        """
        try:
            retval = self.lines[self.cursor]
        except IndexError:
            if not self.remrows:
                return ""
            if self.remvals <= 6:
                nval = min(6, self.remvals)
                self.remrows -= 1
                self.remvals = self.nz 
            else:
                nval = 6
                self.remvals -= nval
            return " {0: .5E}".format(self.const)*nval + "\n"
        else:
            self.cursor += 1
            return retval


def _getline(cube):
    """
    Read a line from cube file.

    First field is an int and the remaining fields are floats.
    
    Parameters
    ----------
    cube : TextIO
        The cubefile from which the line is read.

    Returns
    -------
    line : tuple
        First entry is an int, and the rests are floats.
    """
    line = cube.readline().strip().split()
    return int(line[0]), map(float, line[1:])


def _putline(*args):
    """
    Generate a line to be written to a cube file.

    The first field is an int and the remaining fields are floats.

    Parameters
    ----------
    args : tuple
        First arg is formatted as int and remaining as floats.

    Returns
    -------
    line : string
        Formatted string to be written to file with trailing newline.
    """
    s = "{0:^ 8d}".format(args[0])
    s += "".join("{0:< 12.6f}".format(arg) for arg in args[1:])
    return s + "\n"


def read_cube(fname):
    """
    Read cube file into numpy array.
    
    Parameters
    ----------
    fname : string
        filename of cube file.

    Returns
    -------
    data : numpy.array
        Data from cube file.

    meta : dict
        Meta data from cube file.
    """
    meta = {}
    with open(fname, 'r') as cube:
        # ignore comments
        cube.readline()
        cube.readline()
        natm, meta['org'] = _getline(cube)
        nx, meta['xvec'] = _getline(cube)
        ny, meta['yvec'] = _getline(cube)
        nz, meta['zvec'] = _getline(cube)
        meta['atoms'] = [_getline(cube) for i in range(natm)]
        data = np.zeros((nx*ny*nz))
        idx = 0
        for line in cube:
            for val in line.strip().split():
                data[idx] = float(val)
                idx += 1
    data = np.reshape(data, (nx, ny, nz))
    return data, meta


def read_imcube(rfname, ifname=""):
    """
    Read in two cube files at once.

    One contains the real part and the other contains the
    imag part. If only one filename given, other filename is inferred.
    
    params:

    returns: np.array (real part + j*imag part)

    Parameters
    ----------
    rfname: string
        filename of cube file of real part

    ifname: string
        optional, filename of cube file of imag part

    Returns
    -------
    data : numpy.array
        Data from cube file.

    meta : dict
        Meta data from cube file.
    """
    ifname = ifname or rfname.replace('real', 'imag')
    _debug("reading from files", rfname, "and", ifname)
    re, im = read_cube(rfname), read_cube(ifname)
    fin = np.zeros(re[0].shape, dtype='complex128')
    if re[1] != im[1]:
        _debug("warning: meta data mismatch, real part metadata retained")
    fin += re[0] 
    fin += 1j*im[0]
    return fin, re[1]


def write_cube(data, meta, fname):
    """
    Write volumetric data to cube file.

    Parameters
    ----------
    data: numpy.array
        volumetric data consisting real values

    meta: dict
        dict containing metadata with following keys:

            - atoms: list of atoms in the form (mass, [position])
            - org: origin
            - xvec,yvec,zvec: lattice vector basis

    fname: string
        filename of cubefile (existing files overwritten)
    """
    with open(fname, "w") as cube:
        # first two lines are comments
        cube.write(" Cubefile created by cubetools.py\n  source: none\n")
        natm = len(meta['atoms'])
        nx, ny, nz = data.shape
        cube.write(_putline(natm, *meta['org']))  # 3rd line #atoms and origin
        cube.write(_putline(nx, *meta['xvec']))
        cube.write(_putline(ny, *meta['yvec']))
        cube.write(_putline(nz, *meta['zvec']))
        for atom_mass, atom_pos in meta['atoms']:
            cube.write(_putline(atom_mass, *atom_pos))    # skip the newline
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if (i or j or k) and k % 6 == 0:
                        cube.write("\n")
                    cube.write(" {0: .5E}".format(data[i, j, k]))


def write_imcube(data, meta, rfname, ifname=""):
    """
    Write two cube files from compley valued volumetric data.

    One for the real part and one for the imaginary part.
    Data about atoms, origin and lattice vectors are kept same for both.
    If only one filename given, other filename is inferred.

    Parameters
    ----------
    data: numpy.array
        volumetric data consisting complex values

    meta: dict
        dict containing metadata with following keys:

            - atoms: list of atoms in the form (mass, [position])
            - org: origin
            - xvec,yvec,zvec: lattice vector basis

    rfname: string
        filename of cube file containing real part

    ifname: string
        optional, filename of cube file containing imag part
    """
    ifname = ifname or rfname.replace('real', 'imag')
    _debug("writing data to files", rfname, "and", ifname)
    write_cube(data.real, meta, rfname)
    write_cube(data.imag, meta, ifname)
