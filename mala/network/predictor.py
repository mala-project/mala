"""Tester class for testing a network."""
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import ase.io
import numpy as np
import torch

from mala.common.parameters import printout
from mala.network.runner import Runner


class Predictor(Runner):
    """
    A class for testing a neural network.

    It enables easy inference throughout a test set.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Tester object.

    network : mala.network.network.Network
        Network which is being tested.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the test data.
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
        self.__prepare_to_predict()

    def predict_from_qeout(self, path_to_file):
        """
        Get predicted LDOS for the atomic configuration of a QE.out file.

        Parameters
        ----------
        path_to_file : string
            Path from which to read the atomic configuration.

        Returns
        -------
        predicted_ldos : numpy.array
            Precicted LDOS for these atomic positions.
        """
        atoms = ase.io.read(path_to_file, format="espresso-out")
        return self.predict_for_atoms(atoms)

    def predict_for_atoms(self, atoms):
        """
        Get predicted LDOS for an atomic configuration.

        Parameters
        ----------
        atoms : ase.Atoms
            ASE atoms for which the prediction should be done.

        Returns
        -------
        predicted_ldos : numpy.array
            Precicted LDOS for these atomic positions.
        """
        # Calculate SNAP descriptors.
        snap_descriptors = self.data.descriptor_calculator.\
            calculate_from_atoms(atoms, self.data.grid_dimension)

        # Now reshape and scale the descriptors.
        feature_length = self.data.descriptor_calculator.fingerprint_length
        if self.parameters_full.data.descriptors_contain_xyz:
            snap_descriptors = snap_descriptors[:, :, :, 3:]
            feature_length -= 3

        snap_descriptors = \
            snap_descriptors.astype(np.float32)
        snap_descriptors = \
            snap_descriptors.reshape(
                [self.data.grid_size, feature_length])
        snap_descriptors = \
            torch.from_numpy(snap_descriptors).float()
        snap_descriptors = \
            self.data.input_data_scaler.transform(snap_descriptors)

        # Forward the SNAP descriptors through the network.
        return self.\
               _forward_snap_descriptors(snap_descriptors)

    def _forward_snap_descriptors(self, snap_descriptors):
        """Forwards a scaled tensor of SNAP descriptors through the NN."""
        predicted_outputs = np.zeros((self.data.grid_size,
                                      self.data.target_calculator.\
                                      get_feature_size()))

        for i in range(0, self.number_of_batches_per_snapshot):
            inputs = snap_descriptors[i * self.parameters.mini_batch_size:
                                      (i+1)*self.parameters.mini_batch_size]
            if self.parameters_full.use_gpu:
                inputs = inputs.to('cuda')
            predicted_outputs[i * self.parameters.mini_batch_size:
                                      (i+1)*self.parameters.mini_batch_size] \
                = self.data.output_data_scaler.\
                inverse_transform(self.network(inputs).
                                  to('cpu'), as_numpy=True)

        # Restricting the actual quantities to physical meaningful values,
        # i.e. restricting the (L)DOS to positive values.
        predicted_outputs = self.data.target_calculator.\
            restrict_data(predicted_outputs)
        return predicted_outputs

    # Currently a copy of the prepare_to_test function of the Tester class.
    # Might change in the future.
    def __prepare_to_predict(self):
        """Prepare the tester class to for test run."""
        # We will use the DataSet iterator to iterate over the test data.
        # But since we only want the data per snapshot,
        # we need to make sure the batch size is compatible with that.
        optimal_batch_size = self.\
            _correct_batch_size_for_testing(self.data.grid_size,
                                            self.parameters.mini_batch_size)
        if optimal_batch_size != self.parameters.mini_batch_size:
            printout("Had to readjust batch size from",
                     self.parameters.mini_batch_size, "to",
                     optimal_batch_size)
            self.parameters.mini_batch_size = optimal_batch_size
        self.number_of_batches_per_snapshot = int(self.data.grid_size /
                                                  self.parameters.
                                                  mini_batch_size)
