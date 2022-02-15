"""Interface function for getting Targets."""
from mala.targets.ldos import LDOS
from mala.targets.dos import DOS
from mala.targets.density import Density


def TargetInterface(params):
    """
    Return correct target calculator/parser, based on parameters.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this TargetBase object.

    Returns
    -------
    target : mala.targets.targets.TargetBase or derivative

    """
    if params.targets.target_type == 'LDOS':
        return LDOS(params)
    elif params.targets.target_type == 'DOS':
        return DOS(params)
    elif params.targets.target_type == 'Density':
        return Density(params)
    else:
        raise Exception("Unknown type of target parser requested.")
