from .ldos import LDOS
from .dos import DOS
from .density import Density


def TargetInterface(params):
    """
    Return correct target calculator/parser, based on parameters.

    Parameters
    ----------
    params : fesl.common.parameters.Parameters
        Parameters used to create this TargetBase object.

    Returns
    -------
    target : fesl.targets.targets.TargetBase or derivative

    """
    if params.targets.target_type == 'LDOS':
        return LDOS(params)
    elif params.targets.target_type == 'DOS':
        return DOS(params)
    elif params.targets.target_type == 'Density':
        return Density(params)
    else:
        raise Exception("Unknown type of target parser requested.")
