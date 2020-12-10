'''Class for calculation of SNAP descriptors'''

import ase
import ase.io
from .lammps_utils import *
from lammps import lammps
from .descriptor_base import descriptor_base

class snap(descriptor_base):
    def __init__(self,p):
        super(snap,self).__init__(p)
        self.in_format_ase = ""

    def calculate_from_qe_out(self, qe_out_file, qe_out_directory):
        """Calculates the SNAP descriptors given a QE outfile."""
        self.in_format_ase = "espresso-out"
        return self.calculate_snap(qe_out_directory+qe_out_file,qe_out_directory)

    def calculate_snap(self, infile, outdir):
        """Actual SNAP calculation."""
        lammps_format = "lammps-data"

        # We get the atomic information by using ASE.
        atoms = ase.io.read(infile, format=self.in_format_ase)
        ase_out_path = outdir+"lammps_input.tmp"
        ase.io.write(ase_out_path, atoms, format=lammps_format)

        # We also need to know how big the grid is.
        # Iterating directly through the file is slow, but the
        # grid information is at the top (around line 200).
        if (len(self.dbg_grid_dimensions)==3):
            nx = self.dbg_grid_dimensions[0]
            ny = self.dbg_grid_dimensions[1]
            nz = self.dbg_grid_dimensions[2]
        else:
            qe_outfile = open(files[-1], "r")
            lines = qe_outfile.readlines()
            for line in lines:
                if "FFT dimensions" in line:
                    tmp = line.split("(")[1].split(")")[0]
                    nx = int(tmp.split(",")[0])
                    ny = int(tmp.split(",")[1])
                    nz = int(tmp.split(",")[2])
                    break
        # Build LAMMPS arguments from the data we read.
        lmp_cmdargs = ["-screen", "none", "-log", outdir+"lammps_log.tmp"]
        lmp_cmdargs = set_cmdlinevars(lmp_cmdargs,
            {
            "ngridx":nx,
            "ngridy":ny,
            "ngridz":nz,
            "twojmax":self.parameters.twojmax,
            "rcutfac":self.parameters.rcutfac,
            "atom_config_fname":ase_out_path
            }
        )

        # Build the LAMMPS object.
        lmp = lammps(cmdargs=lmp_cmdargs)

        # An empty string means that the user wants to use the standard input.
        if (self.parameters.lammps_compute_file == ""):
            filepath = __file__.split("snap")[0]
            self.parameters.lammps_compute_file = filepath+"in.bgrid.python"

        # Do the LAMMPS calculation.
        try:
            lmp.file(self.parameters.lammps_compute_file)
        except lammps.LAMMPSException:
            raise Exception("There was a problem during the SNAP calculation. Exiting.")


        # Set things not accessible from LAMMPS
        # First 3 cols are x, y, z, coords
        ncols0 = 3

        # Analytical relation for fingerprint length
        ncoeff = (self.parameters.twojmax+2)*(self.parameters.twojmax+3)*(self.parameters.twojmax+4)
        ncoeff = ncoeff // 24 # integer division
        self.fingerprint_length = ncols0+ncoeff

        # Extract data from LAMMPS calculation.
        snap_descriptors_np = extract_compute_np(lmp, "bgrid", 0, 2, (nz,ny,nx, self.fingerprint_length))

        # switch from x-fastest to z-fastest order (swaps 0th and 2nd dimension)
        snap_descriptors_np = snap_descriptors_np.transpose([2,1,0,3])

        # The next two commands can be used to check whether the calculated SNAP descriptors
        # are identical to the ones calculated with the Sandia workflow. They generally are.
        # print("snap_descriptors_np shape = ",snap_descriptors_np.shape, flush=True)
        # np.save(set[4]+"test.npy", snap_descriptors_np, allow_pickle=True)
        return snap_descriptors_np
