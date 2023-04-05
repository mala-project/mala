"""Tester class for testing a network."""
import ase.io
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np
import torch

from mala.common.parallelizer import printout, get_rank, barrier
from mala.network.runner import Runner


class Predictor(Runner):
    """
    A class for running predictions using a neural network.

    It enables production-level inference.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Predictor object.

    network : mala.network.network.Network
        Network used for predictions.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler, in this case not directly holding data, but serving
        as an interface to Target and Descriptor objects.
    """

    def __init__(self, params, network, data):
        # copy the parameters into the class.
        super(Predictor, self).__init__(params, network, data)
        self.data.grid_dimension = self.parameters.inference_data_grid
        self.data.grid_size = self.data.grid_dimension[0] * \
                              self.data.grid_dimension[1] * \
                              self.data.grid_dimension[2]
        self.test_data_loader = None
        self.number_of_batches_per_snapshot = 0
        self.target_calculator = data.target_calculator

    def predict_from_qeout(self, path_to_file, gather_ldos=False):
        """
        Get predicted LDOS for the atomic configuration of a QE.out file.

        Parameters
        ----------
        path_to_file : string
            Path from which to read the atomic configuration.

        gather_ldos : bool
            Only important if MPI is used. If True, all descriptors
            are gathered on rank 0, and the pass is performed there.
            Helpful for using multiple CPUs for descriptor calculations
            and only one for network pass.

        Returns
        -------
        predicted_ldos : numpy.array
            Precicted LDOS for these atomic positions.
        """
        self.data.grid_dimension = self.parameters.inference_data_grid
        self.data.grid_size = self.data.grid_dimension[0] * \
                              self.data.grid_dimension[1] * \
                              self.data.grid_dimension[2]

        self.data.target_calculator.\
            read_additional_calculation_data(path_to_file, "espresso-out")
        return self.predict_for_atoms(self.data.target_calculator.atoms,
                                      gather_ldos=gather_ldos)

    def predict_for_atoms(self, atoms, gather_ldos=False, temperature=None):
        """
        Get predicted LDOS for an atomic configuration.

        Parameters
        ----------
        atoms : ase.Atoms
            ASE atoms for which the prediction should be done.

        gather_ldos : bool
            Only important if MPI is used. If True, all descriptors
            are gathered on rank 0, and the pass is performed there.
            Helpful for using multiple CPUs for descriptor calculations
            and only one for network pass.

        temperature : float
            If not None, this temperature value will be set in the internal
            target calculator and can be used in subsequent integrations.
            If None, the default temperature loaded from the model will be
            used. Temperature has to be given in K.

        Returns
        -------
        predicted_ldos : numpy.array
            Precicted LDOS for these atomic positions.
        """
        # If there is no inference data grid, we will try to get the grid
        # dimensions from the target calculator, because some data may
        # have been saved there.

        if np.prod(self.parameters.inference_data_grid) > 0:
            self.data.grid_dimension = self.parameters.inference_data_grid
        else:
            # We need to check if we're in size transfer mode.
            old_cell = self.data.target_calculator.atoms.get_cell()
            new_cell = atoms.get_cell()

            # We only need the diagonal elements.
            factor = np.diag(new_cell)/np.diag(old_cell)
            factor = factor.astype(int)
            self.data.grid_dimension = \
                factor * self.data.target_calculator.grid_dimensions

        self.data.grid_size = np.prod(self.data.grid_dimension)

        # Set the tempetature, if necessary.
        if temperature is not None:
            self.data.target_calculator.temperature = temperature

        # Make sure no data lingers in the target calculator.
        self.data.target_calculator.invalidate_target()

        # Calculate descriptors.
        snap_descriptors, local_size = self.data.descriptor_calculator.\
            calculate_from_atoms(atoms, self.data.grid_dimension)

        # Provide info from current snapshot to target calculator.
        self.data.target_calculator.\
            read_additional_calculation_data([atoms, self.data.grid_dimension],
                                             "atoms+grid")
        feature_length = self.data.descriptor_calculator.fingerprint_length

        # The actual calculation of the LDOS from the descriptors depends
        # on whether we run in parallel or serial. In the former case,
        # each batch is forwarded individually (for now), in the latter
        # case, everything is forwarded at once.
        if self.parameters._configuration["mpi"]:
            if gather_ldos is True:
                snap_descriptors = self.data.descriptor_calculator. \
                    gather_descriptors(snap_descriptors)

                # Just entering the forwarding function to wait for the
                # main rank further down.
                if get_rank() != 0:
                    self._forward_snap_descriptors(snap_descriptors, 0)
                    return None

            else:
                if self.data.descriptor_calculator.descriptors_contain_xyz:
                    self.data.target_calculator.local_grid = \
                        snap_descriptors[:, 0:3].copy()
                    self.data.target_calculator.y_planes = \
                        self.data.descriptor_calculator.parameters.\
                        use_y_splitting
                    snap_descriptors = snap_descriptors[:, 6:]
                    feature_length -= 3
                else:
                    raise Exception("Cannot calculate the local grid without "
                                    "calculating the xyz positions of the "
                                    "descriptors. Please revise your "
                                    "script. The local grid is crucial"
                                    " for parallel inference")

                snap_descriptors = \
                    torch.from_numpy(snap_descriptors).float()
                self.data.input_data_scaler.transform(snap_descriptors)
                return self. \
                    _forward_snap_descriptors(snap_descriptors, local_size)

        if get_rank() == 0:
            if self.data.descriptor_calculator.descriptors_contain_xyz:
                snap_descriptors = snap_descriptors[:, :, :, 3:]
                feature_length -= 3

            snap_descriptors = \
                snap_descriptors.reshape(
                    [self.data.grid_size, feature_length])
            snap_descriptors = \
                torch.from_numpy(snap_descriptors).float()
            self.data.input_data_scaler.transform(snap_descriptors)
            return self._forward_snap_descriptors(snap_descriptors)

    def _forward_snap_descriptors(self, snap_descriptors,
                                  local_data_size=None):
        """Forward a scaled tensor of descriptors through the NN."""
        if local_data_size is None:
            local_data_size = self.data.grid_size
        predicted_outputs = \
            np.zeros((local_data_size,
                      self.data.target_calculator.feature_size))

        # Only predict if there is something to predict.
        # Elsewise, we just wait at the barrier down below.
        if local_data_size > 0:
            optimal_batch_size = self.\
                _correct_batch_size_for_testing(local_data_size,
                                                self.parameters.mini_batch_size)
            if optimal_batch_size != self.parameters.mini_batch_size:
                printout("Had to readjust batch size from",
                         self.parameters.mini_batch_size, "to",
                         optimal_batch_size, min_verbosity=0)
                self.parameters.mini_batch_size = optimal_batch_size

            self.number_of_batches_per_snapshot = int(local_data_size /
                                                      self.parameters.
                                                      mini_batch_size)

            for i in range(0, self.number_of_batches_per_snapshot):
                inputs = snap_descriptors[i * self.parameters.mini_batch_size:
                                          (i+1)*self.parameters.mini_batch_size]
                inputs = inputs.to(self.parameters._configuration["device"])
                predicted_outputs[i * self.parameters.mini_batch_size:
                                          (i+1)*self.parameters.mini_batch_size] \
                    = self.data.output_data_scaler.\
                    inverse_transform(self.network(inputs).
                                      to('cpu'), as_numpy=True)

            # Restricting the actual quantities to physical meaningful values,
            # i.e. restricting the (L)DOS to positive values.
            predicted_outputs = self.data.target_calculator.\
                restrict_data(predicted_outputs)
        barrier()
        return predicted_outputs
