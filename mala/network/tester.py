"""Tester class for testing a network."""

import numpy as np

from mala.common.parameters import printout
from mala.network.runner import Runner
from mala.targets.ldos import LDOS
from mala.targets.dos import DOS
from mala.targets.density import Density


class Tester(Runner):
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

    observables_to_test : list
        List of observables to test. Supported are:

            - "ldos": Calculate the MSE loss of the LDOS.
            - "band_energy": Band energy error
            - "band_energy_full": Band energy absolute values (only works with
              list, as both actual and predicted are returned)
            - "total_energy": Total energy error
            - "total_energy_full": Total energy absolute values (only works
              with list, as both actual and predicted are returned)
            - "number_of_electrons": Number of electrons (Fermi energy is not
              determined dynamically for this quantity.
            - "density": MAPE of the density prediction
            - "dos": MAPE of the DOS prediction

    output_format : string
        Can be "list" or "mae". If "list", then a list of results across all
        snapshots is returned. If "mae", then the MAE across all snapshots
        will be calculated and returned.
    """

    def __init__(
        self,
        params,
        network,
        data,
        observables_to_test=["ldos"],
        output_format="list",
    ):
        # copy the parameters into the class.
        super(Tester, self).__init__(params, network, data)
        self.test_data_loader = None
        self.number_of_batches_per_snapshot = 0
        self.observables_to_test = observables_to_test
        self.output_format = output_format
        if self.output_format != "list" and self.output_format != "mae":
            raise Exception("Wrong output format for testing selected.")
        self.target_calculator = data.target_calculator

    def test_all_snapshots(self):
        """
        Test the selected observables for all snapshots.

        Returns
        -------
        results : dict
            A dictionary containing the errors for the selected observables,
            either as list or MAE.
        """
        results = {}
        for observable in self.observables_to_test:
            results[observable] = []

        for i in range(0, self.data.nr_test_snapshots):
            snapshot_result = self.test_snapshot(i)
            for observable in self.observables_to_test:
                results[observable].append(snapshot_result[observable])

        if self.output_format == "list":
            return results

        elif self.output_format == "mae":
            mae_results = {}
            for observable in self.observables_to_test:
                mae_results[observable] = np.mean(np.abs(results[observable]))
            return mae_results

        else:
            raise Exception("Wrong output format for testing selected.")

    def test_snapshot(self, snapshot_number, data_type="te"):
        """
        Test the selected observables for a single snapshot.

        Parameters
        ----------
        snapshot_number : int
            Snapshot which to test.

        data_type : str
            'tr', 'va', or 'te' indicating the partition to be tested

        Returns
        -------
        results : dict
            A dictionary containing the errors for the selected observables.
        """
        actual_outputs, predicted_outputs = self.predict_targets(
            snapshot_number, data_type=data_type
        )

        results = self._calculate_errors(
            actual_outputs,
            predicted_outputs,
            self.observables_to_test,
            snapshot_number,
        )
        return results
    
    def get_energy_targets_and_predictions(self, snapshot_number, data_type="te"):
        """
        Get the energy targets and predictions for a single snapshot.

        Parameters
        ----------
        snapshot_number : int
            Snapshot which to test.

        data_type : str
            'tr', 'va', or 'te' indicating the partition to be tested

        Returns
        -------
        results : dict
            A dictionary containing the errors for the selected observables.
        """
        actual_outputs, predicted_outputs = self.predict_targets(
            snapshot_number, data_type=data_type
        )
        
        energy_metrics = [metric for metric in self.observables_to_test if "energy" in metric]
        targets, predictions = self._calculate_energy_targets_and_predictions(
            actual_outputs,
            predicted_outputs,
            energy_metrics,
            snapshot_number,
        )
        return targets, predictions

    def predict_targets(self, snapshot_number, data_type="te"):
        """
        Get actual and predicted energy outputs for a snapshot.

        Parameters
        ----------
        snapshot_number : int
            Snapshot for which the prediction is done.

        data_type : str
            'tr', 'va', or 'te' indicating the partition to be tested

        Returns
        -------
        actual_outputs : numpy.ndarray
            Actual outputs for snapshot.

        predicted_outputs : numpy.ndarray
            Precicted outputs for snapshot.
        """
        # Adjust batch size and such.
        self.__prepare_to_test(snapshot_number)
        # Make sure no data lingers in the target calculator.
        self.data.target_calculator.invalidate_target()
        # Select the inputs used for prediction
        if data_type == "tr":
            offset_snapshots = 0
            data_set = self.data.training_data_sets[0]
        elif data_type == "va":
            offset_snapshots = self.data.nr_training_snapshots
            data_set = self.data.validation_data_sets[0]
        elif data_type == "te":
            offset_snapshots = (
                self.data.nr_validation_snapshots
                + self.data.nr_training_snapshots
            )
            data_set = self.data.test_data_sets[0]
        else:
            raise ValueError(
                f"Invalid data_type: {data_type} -- Valid options are tr, va, te."
            )
        # Forward through network.
        return self._forward_entire_snapshot(
            offset_snapshots + snapshot_number,
            data_set,
            data_type,
            self.number_of_batches_per_snapshot,
            self.parameters.mini_batch_size,
        )

    def __prepare_to_test(self, snapshot_number):
        """Prepare the tester class to for test run."""
        # We will use the DataSet iterator to iterate over the test data.
        # But since we only want the data per snapshot,
        # we need to make sure the batch size is compatible with that.
        test_snapshot = 0
        grid_size = None
        for snapshot in self.data.parameters.snapshot_directories_list:
            if snapshot.snapshot_function == "te":
                if snapshot_number == test_snapshot:
                    grid_size = snapshot.grid_size
                    break
                test_snapshot += 1

        optimal_batch_size = self._correct_batch_size_for_testing(
            grid_size, self.parameters.mini_batch_size
        )
        if optimal_batch_size != self.parameters.mini_batch_size:
            printout(
                "Had to readjust batch size from",
                self.parameters.mini_batch_size,
                "to",
                optimal_batch_size,
                min_verbosity=0,
            )
            self.parameters.mini_batch_size = optimal_batch_size
        self.number_of_batches_per_snapshot = int(
            grid_size / self.parameters.mini_batch_size
        )
