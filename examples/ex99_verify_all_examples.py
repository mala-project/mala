from ex01_run_singleshot import run_example01
from ex02_hyperparameter_optimization import run_example02
from ex03_resize_snapshots import run_example03
from ex04_snapshot_splitting import run_example04
from ex05_create_snap_descriptors import run_example05
from ex06_postprocessing import run_example06
from ex07_dos_analysis import run_example07
from ex08_training_with_postprocessing import run_example08
from ex09_get_total_energy import run_example09


"""
ex99_verify_all_examples.py: This example confirms whether or not the examples run CORRECTLY. That is, even though they
might run, they may not run correctly, e.g. a network will train but with abysmal performance.  
"""
print("Welcome to FESL.")
print("Running ex99_verify_all_examples.py")

# Example 1: Perform a training.
if run_example01():
    print("Successfully ran ex01_run_singleshot.")
else:
    raise Exception("Ran ex01_run_singleshot but something was off. If you haven't changed any parameters in "
                    "the example, there might be a problem with your installation.")

# Example 2: Perform a hyperparameter optimization.
if run_example02():
    print("Successfully ran ex01_run_singleshot.")
else:
    raise Exception("Ran ex01_run_singleshot but something was off. If you haven't changed any parameters in "
                    "the example, there might be a problem with your installation.")


# Example 3: Resizing of snapshots. Omitted for now, because it depends on data.
try:
    run_example03("/home/fiedlerl/data/Al256/SandiaQE/2.699gcc/")
except:
    print("Could not run example03, did you provide a correct data path?")

# Example 4: Like example 1, but with snapshot splitting.
if run_example04():
    print("Successfully ran ex04_snapshot_splitting.")
else:
    raise Exception("Ran ex04_snapshot_splitting but something was off. If you haven't changed any parameters in "
                    "the example, there might be a problem with your installation.")

# Example 5: SNAP calculation. Ommitted for now, because it depends on LAMMPS installation.
try:
    run_example05()
except:
    print("Could not run example09, most likely because of missing LAMMPS installation.")



# Example 6: Basic postprocessing capabilities.
if run_example06(doplots=False):
    print("Successfully ran ex06_postprocessing.py.")
else:
    raise Exception("Ran ex06_postprocessing but something was off. If you haven't changed any parameters in "
                    "the example, there might be a problem with your installation.")


# Example 7: Density of states analysis.
if run_example07():
    print("Successfully ran ex06_postprocessing.py.")
else:
    raise Exception("Ran ex06_postprocessing but something was off. If you haven't changed any parameters in "
                    "the example, there might be a problem with your installation.")

# Example 8: Train a network, do a prediction and process this prediction.
if run_example08(False, True,doplots=False):
    print("Successfully ran ex08_training_with_postprocessing.py.")
else:
    raise Exception("Ran ex08_training_with_postprocessing but something was off. If you haven't changed any parameters in "
                    "the example, there might be a problem with your installation.")

# Example 9: Calculation of the total energy. Ommitted for now, because it depends on a custom QE installation.
try:
    run_example09()
except:
    print("Could not run example09, most likely because of missing QE installation.")





