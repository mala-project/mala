import mala
from mala import printout
from multiprocessing import Process
import sys
import os
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al36/"


"""
ex10_tensor_board.py: 
to include tensor board link through python script to auto start the tensorboard port.
to run tensorboard from a python script.
to check if the  
"""

def run_example01(desired_loss_improvement_factor=1):
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################

    test_parameters = mala.Parameters()
    # Currently, the splitting in training, validation and test set are
    # done on a "by snapshot" basis. Specify how this is
    # done by providing a list containing entries of the form
    # "tr", "va" and "te".
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]

    # Specify the data scaling.
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    test_parameters.network.layer_activations = ["ReLU"]

    # Specify the training parameters.
    test_parameters.running.max_number_epochs = 20
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"
    test_parameters.running.visualisation = 2
    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################

    data_handler = mala.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                              "Al_debug_2k_nr0.out.npy", data_path,
                              output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                              "Al_debug_2k_nr1.out.npy", data_path,
                              output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                              "Al_debug_2k_nr2.out.npy", data_path,
                              output_units="1/Ry")
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    # The layer sizes can be specified before reading data,
    # but it is safer this way.
    ####################

    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),
                                           100,
                                           data_handler.get_output_dimension()]

    # Setup network and trainer.
    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
    printout("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    printout("Starting training.")
    test_trainer.train_network()
    printout("Training: DONE.")

    ####################
    # RESULTS.
    # Print the used parameters and check whether the loss decreased enough.
    ####################

    printout("Parameters used for this experiment:")
    test_parameters.show()

    if desired_loss_improvement_factor*test_trainer.initial_test_loss\
            < test_trainer.final_test_loss:
        return False
    else:
        return True



    

class TensorboardSupervisor:
    def __init__(self, log_dp):
            self.server = TensorboardServer(log_dp)
            self.server.start()
            print("Started Tensorboard Server")
            self.chrome = ChromeProcess()
            print("Started Chrome Browser")
            self.chrome.start()

    def finalize(self):
        if self.server.is_alive():
            print('Killing Tensorboard Server')
            self.server.terminate()
            self.server.join()


class TensorboardServer(Process):
    def __init__(self, log_dp):
        super().__init__()
        self.os_name = os.name
        self.log_dp = str(log_dp)

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system('tensorboard --logdir="{self.log_dp} --bind_all"')
        elif self.os_name == 'posix':  # Linux
            os.system(f'tensorboard --logdir="{self.log_dp} --bind_all" ')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')
    
    
class ChromeProcess(Process):
    def __init__(self):
        super().__init__()
        self.os_name = os.name
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'start chrome  http://localhost:6006/')
        elif self.os_name == 'posix':  # Linux
            os.system(f'google-chrome http://localhost:6006/')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')

def run_example10():
    tb_sup = TensorboardSupervisor("/home/snehaverma/Downloads/HZDR_work/mala/examples/log_dir")
    tb_sup.finalize()
  ####################
    # RESULTS.
    # Check whether tensorboard has run.
    ####################

    printout("Tensor board has ran succesfully:")

    if run_example10:
        return False
    else: 
        return True


if __name__ == "__main__":
    if run_example01() & run_example10():
        printout("Successfully ran ex10_tensor_board.py.")
    else:
        raise Exception("Ran ex10_tensor_board but something "
                        "was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with "
                        "your installation.")