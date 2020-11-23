'''
For testing purposes it will be helpful to have data that is not DFT / LDOS / FP data.
This class provides an interface to the MNIST dataset that can be used for tests.
The code is copied from https://github.com/MichalDanielDobrzanski/DeepLearningPython35, which
contains code to accompany Micheal Nielsens Book on NN, but ported to Python3.
'''

from .data_base import data_base
import gzip
import pickle
import numpy as np

class data_mockup(data_base):
    def __init__(self, p):
        self.parameters = p.data
        self.training_data_raw = []
        self.validation_data_raw = []
        self.test_data_raw = []
        self.training_data = []
        self.validation_data = []
        self.test_data = []


    def load_data(self):
        """Loads data from the specified directory."""
        f = gzip.open(self.parameters.directory+'mnist.pkl.gz', 'rb')
        self.training_data_raw, self.validation_data_raw, self.test_data_raw = pickle.load(f, encoding="latin1")
        f.close()

    def prepare_data(self):
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
        training_inputs = [np.reshape(x, (784, 1)) for x in self.training_data_raw[0]]
        training_results = [self.vectorized_result(y) for y in self.training_data_raw[1]]
        self.training_data = zip(training_inputs, training_results)
        validation_inputs = [np.reshape(x, (784, 1)) for x in self.validation_data_raw[0]]
        self.validation_data = zip(validation_inputs, self.validation_data_raw[1])
        test_inputs = [np.reshape(x, (784, 1)) for x in self.test_data_raw[0]]
        self.test_data = zip(test_inputs, self.test_data_raw[1])

    def vectorized_result(self,j):
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
