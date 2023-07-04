"""Function to check module availability in MALA."""
import importlib


def check_modules():
    """Check whether/which optional modules MALA can access."""
    # The optional libs in MALA.
    optional_libs = {
        "mpi4py": {"available": False, "description":
                   "Enables inference parallelization."},
        "horovod": {"available": False, "description":
                    "Enables training parallelization."},
        "lammps": {"available": False, "description":
                   "Enables descriptor calculation for data preprocessing "
                   "and inference."},
        "oapackage": {"available": False, "description":
                      "Enables usage of OAT method for hyperparameter "
                      "optimization."},
        "total_energy": {"available": False, "description":
                         "Enables calculation of total energy."},
        "asap3": {"available": False, "description":
                  "Enables trajectory analysis."},
        "dftpy": {"available": False, "description":
                  "Enables OF-DFT-MD initialization."},
        "minterpy": {"available": False, "description":
            "Enables minterpy descriptor calculation for data preprocessing."}
    }

    # Find out if libs are available.
    for lib in optional_libs:
        optional_libs[lib]["available"] = importlib.util.find_spec(lib) \
                                          is not None

    # Print info about libs.
    print("The following optional modules are available in MALA:")
    for lib in optional_libs:
        available_string = "installed" if optional_libs[lib]["available"] \
            else "not installed"
        print("{0}: \t {1} \t {2}".format(lib, available_string,
                                          optional_libs[lib]["description"]))
        optional_libs[lib]["available"] = \
            importlib.util.find_spec(lib) is not None
