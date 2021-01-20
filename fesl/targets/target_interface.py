from .ldos import LDOS
from .dos import DOS
from .density import Density

def TargetInterface(params):
    """Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of target parser.
    Not necessary if preprocessed data is read."""
    if params.targets.target_type == 'LDOS':
        return LDOS(params)
    elif params.targets.target_type == 'DOS':
        return DOS(params)
    elif params.targets.target_type == 'Density':
        return Density(params)
    else:
        raise Exception("Unknown type of target parser requested.")
