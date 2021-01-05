class Snapshot:
    """Represents a snapshot on a hard drive. Handles different ways such a snapshot can be stored."""
    def __init__(self, input_cube_naming_scheme="", input_cube_directory="", input_npy_file="", input_npy_directory="",
                 input_units="", output_qe_out_file="", output_qe_out_directory="", output_npy_file="",
                 output_npy_directory="", output_units=""):
        # Inputs.
        self.input_cube_naming_scheme = input_cube_naming_scheme
        self.input_cube_directory = input_cube_directory
        self.input_npy_file = input_npy_file
        self.input_npy_directory = input_npy_directory
        self.input_units = input_units

        # Outputs.
        self.output_qe_out_file = output_qe_out_file
        self.output_qe_out_directory = output_qe_out_directory
        self.output_npy_file = output_npy_file
        self.output_npy_directory = output_npy_directory
        self.output_units = output_units

    @classmethod
    def as_npy_npy(cls, input_npy_file, input_npy_directory, output_npy_file, output_npy_directory,
                   input_units, output_units):
        """
        A snapshot consisting of npy files for both input and output files.
            input_npy_file: File name for inputs.
            input_npy_directory: Directory for inputs.
            output_npy_file: File name for outputs.
            output_npy_directory: Directory for outputs.
            input_units: Units for inputs (default: "None").
            output_units: Units for outputs (default: "eV").
        """

        return cls(input_npy_file=input_npy_file, input_npy_directory=input_npy_directory,
                   output_npy_file=output_npy_file, output_npy_directory=output_npy_directory,
                   input_units=input_units, output_units=output_units)

    @classmethod
    def as_qeout_cube(cls, input_cube_naming_scheme, input_cube_directory, output_qe_out_file, output_qe_out_directory,
                      input_units, output_units):
        """
        A snapshot consisting of multiple cube files for input and a QE.out file for output.
            input_cube_naming_scheme: Naming scheme for input cube files.
            input_cube_directory: Directory for inputs.
            output_qe_out_file: File name for outputs.
            output_qe_out_directory: Directory for outputs.
            input_units: Units for inputs (default: "None").
            output_units: Units for outputs (default: "eV").
        """

        return cls(input_cube_naming_scheme=input_cube_naming_scheme, input_cube_directory=input_cube_directory,
                   output_qe_out_file=output_qe_out_file, output_qe_out_directory=output_qe_out_directory,
                   input_units=input_units, output_units=output_units)
