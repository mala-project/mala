import mala
from mala import printout
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
data_path = "/home/fiedlerl/jobs/qe/Be2/data_for_qgp/"

"""
WORK IN PROGRESS

Use Bayesian optimization in MALA.
The code structure is not optimal and only for discussion purposes. 
Once we have converged to a common goal, all of this will be implemented
rigourosly. 
"""

model_to_optimize = None

def myf(predicted):
    predicted = fun(input)
    predicted = np.reshape(predicted, (1,np.product(predicted.shape)))[0].tolist()
    print("Function desired: "+str(desired))
    print("Function predicted: "+str(predicted))
    difference = np.linalg.norm((desired - predicted),2)**2
    print("Difference: "+str(difference))
    return difference


# Use Bayesian optimization to train a model, in this case a Gaussian Process.
def bayesian_optimization_for_model_training():
    params = mala.Parameters()

    # Specify the data scaling.
    params.data.input_rescaling_type = "feature-wise-standard"
    params.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    params.model.loss_function_type = "gaussian_likelihood"

    # Specify the training parameters.
    params.running.max_number_epochs = 10

    # This should be 1, and MALA will set it automatically to, if we don't.
    params.running.mini_batch_size = 40
    params.running.learning_rate = 0.1
    params.running.trainingtype = "Adam"
    params.targets.target_type = "Density"
    #params.debug.grid_dimensions = [10, 10, 1]
    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################

    data_handler = mala.DataHandler(params)
    inputs_folder = data_path+"inputs_snap/"
    outputs_folder = data_path+"outputs_density/"
    additional_folder = data_path+"additional_info_qeouts/"
    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                              "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
    data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                              "snapshot1.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
    data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                              "snapshot2.out.npy", outputs_folder, add_snapshot_as="te",
                              output_units="None", calculation_output_file=additional_folder+"snapshot2.out")
    data_handler.prepare_data(transpose_data=True)
    printout("Read data: DONE.")

    ####################
    # MODEL SETUP
    # Set up the model and trainer we want to use.
    ####################
    params.model.kernel = "linear"

    # Hacky, on
    global model_to_optimize
    model_to_optimize = mala.GaussianProcesses(params, data_handler)
    max_iter = 10
    for i in range(max_iter):
        print("Iteration" + str(i))
        myProblem = GPyOpt.methods.BayesianOptimization(myf, bounds,
                                                        acquisition_type='EI')
        myProblem.run_optimization(i)

    # Instead of mala Trainer class, use Bayesian optimization.
    bounds = [
        {'electronic density': 'E_D', 'type': 'continuous', 'domain': (0, 1)}]

    ####################
    # TESTING
    # Pass the first test set snapshot (the test snapshot).
    ####################

    tester = mala.Tester(params, model, data_handler)
    actual_density, predicted_density = tester.test_snapshot(0)
    # First test snapshot --> 2nd in total
    data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(2))
    actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
    predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
    printout(actual_number_of_electrons, predicted_number_of_electrons)
