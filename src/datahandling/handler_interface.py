from .handler_qeout_ldoscube import handler_qeout_ldoscube
from .handler_mnist import handler_mnist


def handler_interface(params, descriptor_calculator = None):
    '''Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of data handler.'''
    if params.data.datatype_in == 'qe.out' and params.data.datatype_out == 'ldos.cube':
        if (descriptor_calculator is not None ):
            return handler_qeout_ldoscube(params, descriptor_calculator)
        else:
            raise Exception("Using QuantumEspresso outfiles as input requires the definition of a descriptor calcualtor.")
    if params.data.datatype_in == 'mnist' and params.data.datatype_out == 'mnist':
        return handler_mnist(params)
    else:
        raise Exception("Unknown combination of input and output data requested.")
