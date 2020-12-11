'''
For testing purposes it will be helpful to have data that is not DFT / LDOS / FP data.
This class provides an interface to the MNIST dataset that can be used for tests.
The code is copied from https://github.com/MichalDanielDobrzanski/DeepLearningPython35, which
contains code to accompany Micheal Nielsens Book on NN, but ported to Python3.
'''

from .HanderBase import HandlerBase
import gzip
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset


class handler_mnist(HandlerBase):
    """Mock-Up data class that is supposed to work like the actual data classes going to be used for FP/LDOS data,
    but operates on the MNIST data."""
    def __init__(self, p):
        super(handler_npy,self).__init__(p)

    def load_data(self):
        """Loads data from the specified directory."""
        f = gzip.open(self.parameters.directory + 'mnist.pkl.gz', 'rb')
        self.training_data_raw, self.validation_data_raw, self.test_data_raw = pickle.load(
            f, encoding="latin1")
        f.close()

    def get_input_dimension(self):
        """Returns the dimension of the input vector."""
        return 784

    def get_output_dimension(self):
        """Returns the dimension of the output vector."""
        return 10

    def prepare_data(self, less_training_pts=0, less_validation_points=0, less_test_points=0):
        """Prepares the data to be used in an ML workflow:
        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
        containing the input image.  ``y`` is a 10-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x``.
        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
        numpy.ndarry containing the input image, and ``y`` is the
        corresponding classification, i.e., the digit values (integers)
        corresponding to ``x``.
        Obviously, this means we're using slightly different formats for
        the training data and the validation / test data.  These formats
        turn out to be the most convenient for use in our neural network
        code."""
        self.training_inputs = torch.from_numpy(np.squeeze([np.reshape(x, (784, 1))
                            for x in self.training_data_raw[0]])).float()
        self.training_results = torch.from_numpy(np.squeeze([self.vectorized_result(
            y) for y in self.training_data_raw[1]])).float()
        if (less_training_pts != 0):
            self.training_inputs.resize_(less_training_pts, self.get_input_dimension())
            self.training_results.resize_(less_training_pts, self.get_output_dimension())

        self.validation_inputs = torch.from_numpy(np.squeeze([np.reshape(x, (784, 1))
                            for x in self.validation_data_raw[0]])).float()
        self.validation_results = torch.from_numpy(np.squeeze([self.vectorized_result(
            y) for y in self.validation_data_raw[1]])).float()
        if (less_validation_points != 0):
            self.validation_inputs.resize_(less_validation_points, self.get_input_dimension())
            self.validation_results.resize_(less_validation_points, self.get_output_dimension())

        self.test_inputs = torch.from_numpy(np.squeeze([np.reshape(x, (784, 1))
                            for x in self.test_data_raw[0]])).float()
        self.test_results = torch.from_numpy(np.squeeze([self.vectorized_result(
            y) for y in self.test_data_raw[1]])).float()
        if (less_test_points != 0):
            self.test_inputs.resize_(less_test_points, self.get_input_dimension())
            self.test_results.resize_(less_test_points, self.get_output_dimension())


        self.training_data_set = TensorDataset(self.training_inputs, self.training_results)
        self.validation_data_set = TensorDataset(self.validation_inputs, self.validation_results)
        self.test_data_set = TensorDataset(self.test_inputs, self.test_results)

        self.nr_training_data = (self.training_inputs.size())[0]
        self.nr_validation_data = (self.validation_inputs.size())[0]
        self.nr_test_data = (self.test_inputs.size())[0]

    def vectorized_result(self, j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

if __name__ == "__main__":
    raise Exception(
        "data_mockup.py - test of basic functions not yet implemented.")
