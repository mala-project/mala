from .network import Network
from .dummy_model import DummyModel
from mala import Parameters


def ModelInterface(params: Parameters):
    if params.model.type == "feed-forward":
        return Network
    elif params.model.type == "dummy":
        return DummyModel
    else:
        raise Exception("Unknown model type.")
