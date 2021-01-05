from .handler_qeout_cube import HandlerQEoutCube
from .handler_mnist import handler_mnist
from .handler_npy import HandlerNpy
from .data_scaler import DataScaler
from fesl.descriptors.descriptor_interface import DescriptorInterface
from fesl.targets.target_interface import TargetInterface


def HandlerInterface(params, input_data_scaler=None, output_data_scaler=None, target_parser=None,
                     descriptor_calculator=None):
    """Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of data handler."""

    # We ALWAYS require a scaling object.
    # If no scaling is requested by the user the scaler will simply do nothing.

    if input_data_scaler is None:
        input_data_scaler = DataScaler(params.data.input_rescaling_type)
    if output_data_scaler is None:
        output_data_scaler = DataScaler(params.data.output_rescaling_type)
    if target_parser is None:
        target_parser = TargetInterface(params)
    if descriptor_calculator is None:
        descriptor_calculator = DescriptorInterface(params)

    handler = None

    # QuantumEspresso outfiles and *.cube files
    if params.data.datatype_in == 'qe.out' and params.data.datatype_out == '*.cube':
        handler = HandlerQEoutCube(params, input_data_scaler, output_data_scaler, descriptor_calculator, target_parser)

    # Saved numpy arrays.
    if params.data.datatype_in == '*.npy' and params.data.datatype_out == '*.npy':
        handler = HandlerNpy(params, input_data_scaler, output_data_scaler, descriptor_calculator, target_parser)

    # MNIST data.
    if params.data.datatype_in == 'mnist' and params.data.datatype_out == 'mnist':
        handler = handler_mnist(params, input_data_scaler, output_data_scaler)

    if handler is not None:
        return handler
    else:
        raise Exception("Unknown combination of input and output data requested.")
