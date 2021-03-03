from fesl.common.parameters import printout
from ex01_run_singleshot import run_example01
from ex02_preprocess_data import run_example02
from ex03_postprocess_data import run_example03
from ex04_hyperparameter_optimization import run_example04
from ex05_training_with_postprocessing import run_example05
from ex06_advanced_hyperparameter_optimization import run_example06
from ex07_checkpoint_training import run_example07
from ex08_checkpoint_hyperopt import run_example08

"""
ex99_verify_all_examples.py: This example confirms whether or not the examples
run CORRECTLY. That is, even though they might run, they may not run correctly,
e.g. a network will train but with abysmal performance.  
"""
printout("Welcome to FESL.")
printout("Running ex99_verify_all_examples.py")

# Example 1: Perform a training.
if run_example01():
    printout("Successfully ran ex01_run_singleshot.")
else:
    raise Exception("Ran ex01_run_singleshot but something was off. If you "
                    "haven't changed any parameters in "
                    "the example, there might be a problem with your "
                    "installation.")

# Example 2: Preprocess data.
try:
    if run_example02():
        printout("Successfully ran ex02_preprocess_data.")
except ModuleNotFoundError:
    printout("Could not run ex02_preprocess_data, most likely because of"
             "missing LAMMPS installation.")

# Example 3: Postprocess data. Run it twice, once with Quantum Espresso.
if run_example03(do_total_energy=False):
    printout("Successfully ran ex03_postprocess_data.")
else:
    raise Exception("Ran ex03_postprocess_data but something was off."
                    " If you haven't changed any parameters in "
                    "the example, there might be a problem with your"
                    " installation.")
try:
    if run_example03(do_total_energy=True):
        printout("Successfully ran ex03_postprocess_data.")
except ModuleNotFoundError:
    printout("Could not run ex03_postprocess_data, most likely because of "
             "missing QE installation.")


# Example 4: Perform Hyperparameter optimization.
if run_example04():
    printout("Successfully ran ex04_hyperparameter_optimization.py.")
else:
    raise Exception("Ran ex04_hyperparameter_optimization but something "
                    "was off. If you haven't changed any parameters in "
                    "the example, there might be a problem with "
                    "your installation.")

# Example 5: Train a network, do a prediction and process this prediction.
if run_example05(True, True):
    printout("Successfully ran ex05_training_with_postprocessing.py.")
else:
    raise Exception("Ran ex05_training_with_postprocessing but something "
                    "was off. If you haven't changed any parameters in "
                    "the example, there might be a problem with your "
                    "installation.")

# Example 10: Novel hyperparameter optimization techniques.
if run_example06():
    printout("Successfully ran ex06_advanced_hyperparameter_optimization.py.")
else:
    raise Exception(
        "Ran ex10_advanced_hyperparameter_optimization but something was off."
        " If you haven't changed any parameters in "
        "the example, there might be a problem with your installation.")


if run_example07():
    printout("Successfully ran ex07_checkpoint_training.")
else:
    raise Exception("Ran ex07_checkpoint_training but something was off."
                    " If you haven't changed any parameters in "
                    "the example, there might be a problem with your"
                    " installation.")

if run_example08():
    printout("Successfully ran ex08_checkpoint_hyperopt.")
else:
    raise Exception("Ran ex08_checkpoint_hyperopt but something was off."
                    " If you haven't changed any parameters in "
                    "the example, there might be a problem with your"
                    " installation.")
