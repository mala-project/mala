from .snap import snap


def descriptor_interface(params):
    '''Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of descriptor calculator.'''
    if params.descriptors.descriptor_type == 'SNAP':
        return snap(params)
    else:
        raise Exception("Unknown type of descriptor calculator requested.")
