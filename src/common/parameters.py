'''
parameters.py: Includes several classes that structure the parameters used within this framework.
An object of the central parameters() class is the only thing you need to get start training (except from the data of course).
'''


# Subclasses that make up the final parameters class.

class parameters_base():
    def __init__(self):
        raise exception("Error: No constructor implemented.")

    def show(self, indent=""):
        for v in vars(self):
                print(indent+'%-15s: %s' % (v, getattr(self, v)))

class parameters_network(parameters_base):
    """Neural Network parameter subclass."""
    def __init__(self):
        self.nn_type = "feed-forward"

class parameters_descriptors(parameters_base):
    """Fingerprint descriptor parameter subclass."""
    def __init__(self):
        self.descriptor_type = "SNAP"

class parameters_data(parameters_base):
    """Dataset interface parameter subclass."""
    def __init__(self):
        self.datatype = "FP+LDOS"
        self.directory ="~/data"


# TODO: Add keyword arguments that allow for a passing of the arguments in the constructor.

class parameters():
    """Parameter class for ML-DFT@CASUS."""
    def __init__(self):
        self.comment = 0
        '''Comment a used parameter dataset.'''
        self.network = parameters_network()
        self.descriptors = parameters_descriptors()
        self.data = parameters_data()





    def show(self):
        """Prints all the parameters bundled in this class."""
        print("--- "+self.__doc__+" ---")
        for v in vars(self):
            if (isinstance(getattr(self, v), parameters_base)):
                parobject = getattr(self, v)
                print("--- "+parobject.__doc__+" ---")
                parobject.show("\t")
            else:
                print('%-15s: %s' % (v, getattr(self, v)))

if __name__ == "__main__":
    p = parameters()
    p.show()
