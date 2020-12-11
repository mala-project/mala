"""Interface for descriptor calculators. A descriptor calculator is used
to encode information about the chemical neighborhood on the grid. (e.g.: SNAP)"""


from .snap import SNAP


def descriptor_interface(params):
    '''Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of descriptor calculator.'''
    if params.descriptors.descriptor_type == 'SNAP':
        return SNAP(params)
    else:
        raise Exception("Unknown type of descriptor calculator requested.")
