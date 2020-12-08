"""Interface for a target parser. The target parser reads a certain physical
quantity (e.g. the LDOS) from a specified input format and performs, if necessary,
calculations based on this raw input."""

from .ldos import ldos

def target_interface(params):
    '''Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of descriptor calculator.'''
    if params.targets.target_type == 'LDOS':
        return ldos(params)
    else:
        raise Exception("Unknown type of target parser requested.")
