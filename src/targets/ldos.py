'''Local density of states.'''

from .cube_parser import read_cube
from .target_base import TargetBase
import glob
import math

class LDOS(TargetBase):
    def __init__(self,p):
        super(LDOS, self).__init__(p)
        self.target_length = self.parameters.ldos_gridsize

    def read_from_cube(self, file_name_scheme, dir):
        """Reads the LDOS data from multiple cube files located in the snapshot directory."""

        # First determine how many digits the last file in the list of LDOS.cube files
        # will have.
        # QuantumEspresso saves the pp.x cube files like this:
        # tmp.pp001ELEMENT_ldos.cube
        # tmp.pp002ELEMENT_ldos.cube
        # tmp.pp003ELEMENT_ldos.cube
        # ...
        # tmp.pp100ELEMENT_ldos.cube

        digits = int(math.log10(self.parameters.ldos_gridsize))+1

        # Iterate over the amount of specified LDOS input files.
        # QE is a Fortran code, so everything is 1 based.
        for i in range(1,self.parameters.ldos_gridsize+1):
            tmp_file_name = file_name_scheme
            tmp_file_name = tmp_file_name.replace("*",str(i).zfill(digits))

            # Open the cube file
            data, meta = read_cube(dir+tmp_file_name)
            print(data)
            quit()

        # print(dir+file_name_scheme)
        # ldosfilelist = glob.glob(dir+file_name_scheme)
        # print(ldosfilelist)
