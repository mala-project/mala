from fesl.common.parameters import Parameters
from fesl.targets.dos import DOS
import numpy as np
from ase.units import Rydberg

"""
ex07_dos_analysis.py: Shows how the FESL code can be used to further process the DOS. 
"""


def dos_analysis(dos, dos_data, integration, ref):
    # DFT values.

    nr_electrons_dft = dos.get_number_of_electrons(dos_data, integration_method=integration)
    e_band_dft = dos.get_band_energy(dos_data, integration_method=integration)
    fermi_dft = dos.fermi_energy_eV

    # self-consistent values.
    fermi_sc = dos.get_self_consistent_fermi_energy_ev(dos_data, integration_method=integration)
    nr_electrons_sc = dos.get_number_of_electrons(dos_data, integration_method=integration, fermi_energy_eV=fermi_sc)
    e_band_sc = dos.get_band_energy(dos_data, integration_method=integration, fermi_energy_eV=fermi_sc)

    print("Used integration method: ", integration)
    print(
        "Fermi energy source\t#Electrons\tE_band[eV]\tE_Fermi[eV]\tError_E_band[eV]\tError_E_band[meV/atom]\tE_band("
        "DFT)[eV]")
    print("DFT", "\t", nr_electrons_dft, "\t", e_band_dft, "\t", fermi_dft, "\t", e_band_dft - ref, "\t",
          (e_band_dft - ref) * (1000 / 256), "\t", ref)
    print("SC", "\t", nr_electrons_sc, "\t", e_band_sc, "\t", fermi_sc, "\t", e_band_sc - ref, "\t",
          (e_band_sc - ref) * (1000 / 256), "\t", ref)


print("Welcome to FESL.")
print("Running ex07_dos_analysis.py")

# This is done manually at the moment.
eband_exact_rydberg = 737.79466828 + 4.78325172 - 554.11311050
eband_exact_ev = eband_exact_rydberg * Rydberg

####################
# PARAMETERS
# All parameters are handled from a central parameters class that contains subclasses.
####################
test_parameters = Parameters()
test_parameters.targets.ldos_gridsize = 250
test_parameters.targets.ldos_gridspacing_ev = 0.1
test_parameters.targets.ldos_gridoffset_ev = -10.0

# Create a DOS object and provide additional parameters.
dos = DOS(test_parameters)
dos.read_additional_calculation_data("qe.out", "./data/QE_Al.scf.pw.out")

# Load a precalculated DOS file.
dos_data = np.load("./data/Al_DOS_nr0.npy")

dos_analysis(dos, dos_data, "analytical", eband_exact_ev)

print("Successfully ran ex07_dos_analysis.py.")
print("Parameters used for this experiment:")
test_parameters.show()
