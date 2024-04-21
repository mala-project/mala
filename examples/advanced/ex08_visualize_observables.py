import os

import mala

from mala.datahandling.data_repo import data_repo_path

atoms_path = os.path.join(
    os.path.join(data_repo_path, "Be2"), "Be_snapshot1.out"
)
ldos_path = os.path.join(
    os.path.join(data_repo_path, "Be2"), "Be_snapshot1.out.npy"
)
"""
Shows how MALA can be used to visualize observables of interest. 
"""

####################
# 1. READ ELECTRONIC STRUCTURE DATA
# This data may be read as part of an ML-DFT model inference.
# In this case, the LDOS calculator can be taken directly from the
# Predictor/ASE calculator class. The atom object does not have to be read
# in, since it is already available from the prediction, as are the LDOS
# parameters.
####################

parameters = mala.Parameters()
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5

ldos_calculator = mala.LDOS.from_numpy_file(parameters, ldos_path)
ldos_calculator.read_additional_calculation_data(atoms_path)

####################
# 2. CALCULATE AND VISUALIZE OBSERVABLES
# With these functions, observables of interest can be visualized.
####################

# The DOS can be visualized on the correct energy grid.
density_of_states = ldos_calculator.density_of_states
energy_grid = ldos_calculator.energy_grid

# The density can be saved into a .cube file for visualization with standard
# electronic structure visualization software.
density_calculator = mala.Density.from_ldos_calculator(ldos_calculator)
density_calculator.write_to_cube("Be_density.cube")

# The radial distribution function can be visualized on discretized radii.
rdf, radii = ldos_calculator.radial_distribution_function_from_atoms(
    ldos_calculator.atoms, number_of_bins=500
)

# The static structure factor can be visualized on a discretized k-grid.
static_structure, kpoints = ldos_calculator.static_structure_factor_from_atoms(
    ldos_calculator.atoms, number_of_bins=500, kMax=12
)
