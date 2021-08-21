import torch
from .model import Model


class DummyModel(Model):
    """
    Shows how a custom, non-neural network model can be implemented into MALA.

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

    def forward(self, inputs):
        return torch.sum(inputs)*self.factor

    def calculate_loss(self, output, target):
        return output - target

    def tune_model(self, loss, step_width):
        self.factor = self.factor - (loss*step_width)
