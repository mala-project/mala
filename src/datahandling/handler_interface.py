from .handler_qeout_cube import HandlerQEoutCube
from .handler_mnist import handler_mnist
from .handler_npy import HandlerNpy


def HandlerInterface(params, descriptor_calculator=None, target_parser=None, data_scaler=None):
    """Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of data handler."""

    # We ALWAYS require a scaling object.
    # If no scaling is requested by the user the scaler will simply do nothing.

    if data_scaler is None:
        raise Exception("No data scaler was specified.")

    #################################
    # Handlers that require descriptor calculators
    # and/or target parsers.
    #################################

    # QuantumEspresso outfiles and *.cube files
    if params.data.datatype_in == 'qe.out' and params.data.datatype_out == '*.cube':
        if descriptor_calculator is None:
            raise Exception("Using QuantumEspresso outfiles as input requires the definition of a descriptor "
                            "calcualtor.")
        elif target_parser is None:
            raise Exception("Using cube files as input requires the definition of a target parsers.")
        else:
            return HandlerQEoutCube(params, descriptor_calculator, target_parser, data_scaler)

    #################################
    # Handlers that read data as is.
    #################################

    # Saved numpy arrays.
    if params.data.datatype_in == '*.npy' and params.data.datatype_out == '*.npy':
        return HandlerNpy(params, data_scaler)

    # MNIST data.
    if params.data.datatype_in == 'mnist' and params.data.datatype_out == 'mnist':
        return handler_mnist(params, data_scaler)

    # There is no point in proceeding without data.
    else:
        raise Exception("Unknown combination of input and output data requested.")
