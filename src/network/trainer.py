'''
trainer.py contains the training class used to train a network.
This could be absorbed into the network.py class and maybe it will be at some point
or serve as a parent class for multiple trainers, should need be.
'''

class trainer():
    """A class for training a neural network."""
    def __init__(self, p):
        # copy the parameters into the class.
        self.parameters = p.training

    def train_network(self, network, data):
        """Given a network and data, this network is trained on this data."""
        for epoch in range(self.max_number_epochs):
            

    def validate_network(self, network, data)


if __name__ == "__main__":
    raise Exception(
        "trainer.py - test of basic functions not yet implemented.")
