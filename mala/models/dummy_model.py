import torch
import torch.nn
from .model import Model
import pickle


class DummyModel(Model):
    """
    Shows how a custom, non-neural models model can be implemented into MALA.

    This still uses pytorch for infrastructure. The relevant functions have
    to be implemented. The "model" currently only works with the electronic
    density. The "model" will simply sum the inputs, multiply them by a
    factor and be done. This factor can be tuned after each pass.
    This "model" is of course no actual model and not capable of meaningful
    predictions in the slightest. It just shows which functions would
    have to be implemented where.
    """

    def __init__(self, params):
        super(DummyModel, self).__init__(params)
        self.factor = 1

        # This is a dummy for now. If we decide to use pytorch optimizer
        # for optimization of non NN models, it will have to be filled.
        self.dummy_layer = torch.nn.Linear(10, 10)

    def forward(self, inputs):
        return torch.sum(inputs)*self.factor

    def calculate_loss(self, output, target):
        return (output - target).mean()

    def tune_model(self, loss, step_width):
        self.factor = self.factor - (loss*step_width)

    def save_model(self, path_to_file):
        """
        Save the model to file

        This function serves as an interfaces to pytorchs own saving
        functionalities AND possibly own saving needs.

        Parameters
        ----------
        path_to_file : string
            Path to the file in which the models should be saved.
        """
        with open(path_to_file, 'wb') as handle:
            pickle.dump(self.factor, handle, protocol=4)

    @classmethod
    def load_from_file(cls, params, path_to_file):
        """
        Load the model from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the models should be created.
            Has to be compatible to the models architecture. This is usually
            enforced by using the same Parameters object (and saving/loading
            it to)

        path_to_file : string
            Path to the file from which the models should be loaded.

        Returns
        -------
        loaded_network : Network
            The models that was loaded from the file.
        """
        with open(path_to_file, 'rb') as handle:
            new_factor = pickle.load(handle)
        new_model = DummyModel(params)
        new_model.factor = new_factor
        return new_model
