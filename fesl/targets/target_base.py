class TargetBase:
    """Base class for a target quantity parser. Target parsers read the target quantity
    (i.e. the quantity the NN will learn to predict) from a specified file format."""
    def __init__(self, p):
        self.parameters = p.targets
        self.target_length = 0

    def read_from_cube(self):
        raise Exception("No function defined to read this quantity from a .cube file.")

    def get_density(self):
        raise Exception("No function to calculate or provide the density has been implemented for this target type.")

    def get_density_of_states(self):
        raise Exception("No function to calculate or provide the density of states (DOS) has been implemented for this target type.")

    def get_band_energy(self):
        raise Exception("No function to calculate or provide the band energy has been implemented for this target type.")

    @staticmethod
    def convert_units(array, in_units="eV"):
        raise Exception("No unit conversion method implemented for this target type.")

    @staticmethod
    def backconvert_units(array, out_units):
        raise Exception("No unit back conversion method implemented for this target type.")