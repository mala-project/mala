'''
Base class for all data objects.
'''


class data_base():
    """Base class for all data objects."""

    def __init__(self):
        self.training_data_set = []
        self.validation_data_set = []
        self.test_data_set = []

        self.nr_training_data = 0
        self.nr_test_data = 0
        self.nr_validation_data = 0

    def load_data(self):
        """Loads data from the specified directory."""
        raise Exception("load_data not implemented.")

    def prepare_data(self):
        """Prepares the data to be used in an ML workflow (i.e. splitting it in training, validation, test data)"""
        raise Exception("prepare_data not implemented.")

    def save_prepared_data(self):
        """Saves the data after it was altered/preprocessed by the ML workflow. E.g. SNAP descriptor calculation."""
        raise Exception("save_data not implemented.")

    def get_input_dimension(self):
        """Returns the dimension of the input vector."""
        raise Exception("prepare_data not implemented.")

    def get_output_dimension(self):
        """Returns the dimension of the output vector."""
        raise Exception("prepare_data not implemented.")

    def dbg_reduce_number_of_data_points(self, newnumber):
        raise Exception("dbg_reduce_number_of_data_points not implemented.")




if __name__ == "__main__":
    d = data_base()
    print("data_base.py - basic functions are working.")
