"""Interface for objective functions for hyperparameter optimization."""


from .objective_feedforward import ObjectiveFeedForward


def ObjectiveInterface(params, datahandler):
    """Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of descriptor calculator."""
    if params.network.nn_type == "feed-forward":
        return ObjectiveFeedForward(params, datahandler)
    else:
        raise Exception("Unknown type of hyperparameter optimization objective requested.")
