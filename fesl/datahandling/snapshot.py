class Snapshot:
    """Represents a snapshot on a hard drive. Handles different ways such a snapshot can be stored."""
    def __init__(self, input_npy_file="", input_npy_directory="",
                 input_units="",  output_npy_file="",
                 output_npy_directory="", output_units="", calculation_output=""):
        # Inputs.
        self.input_npy_file = input_npy_file
        self.input_npy_directory = input_npy_directory
        self.input_units = input_units

        # Outputs.
        self.output_npy_file = output_npy_file
        self.output_npy_directory = output_npy_directory
        self.output_units = output_units

        # Calculation output.
        self.calculation_output = calculation_output