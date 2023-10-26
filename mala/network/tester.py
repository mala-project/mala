"""Tester class for testing a network."""
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
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

    def __init__(self, params, network, data, observables_to_test=["ldos"],
                 output_format="list"):
        # copy the parameters into the class.
        super(Tester, self).__init__(params, network, data)
        self.test_data_loader = None
        self.number_of_batches_per_snapshot = 0
        self.observables_to_test = observables_to_test
        self.output_format = output_format
        if self.output_format != "list" and self.output_format == "mae":
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

    def test_snapshot(self, snapshot_number, data_type='te'):
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
        actual_outputs, predicted_outputs = \
            self.predict_targets(snapshot_number, data_type=data_type)

        results = {}
        for observable in self.observables_to_test:
            try:
                results[observable] = self.\
                    __calculate_observable_error(snapshot_number,
                                                observable, predicted_outputs,
                                                actual_outputs)
            except ValueError as e:
                printout(f"Error calculating observable: {observable} for snapshot {snapshot_number}",  min_verbosity=0)
                printout(e, min_verbosity=2)
                results[observable] = np.inf
        return results

    def predict_targets(self, snapshot_number, data_type='te'):
        """
        Get actual and predicted output for a snapshot.

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
        if data_type == 'tr':
            offset_snapshots = 0
            data_set = self.data.training_data_sets[0]
        elif data_type == 'va':
            offset_snapshots = self.data.nr_training_snapshots
            data_set = self.data.validation_data_sets[0]
        elif data_type == 'te':
            offset_snapshots = self.data.nr_validation_snapshots + \
                               self.data.nr_training_snapshots
            data_set = self.data.test_data_sets[0]
        else:
            raise ValueError(f"Invalid data_type: {data_type} -- Valid options are tr, va, te.")
        # Forward through network.
        return self.\
            _forward_entire_snapshot(offset_snapshots+snapshot_number,
                                     data_set,
                                     data_type,
                                     self.number_of_batches_per_snapshot,
                                     self.parameters.mini_batch_size)

    def __calculate_observable_error(self, snapshot_number, observable,
                                     predicted_target, actual_target):
        if observable == "ldos":
            return np.mean((predicted_target - actual_target)**2)

        elif observable == "band_energy":
            target_calculator = self.data.target_calculator
            if not isinstance(target_calculator, LDOS) and not \
                    isinstance(target_calculator, DOS):
                raise Exception("Cannot calculate the band energy from this "
                                "observable.")
            target_calculator.\
                read_additional_calculation_data(
                self.data.get_snapshot_calculation_output(snapshot_number))

            target_calculator.read_from_array(actual_target)
            actual = target_calculator.band_energy

            target_calculator.read_from_array(predicted_target)
            predicted = target_calculator.band_energy
            return actual - predicted

        elif observable == "band_energy_full":
            target_calculator = self.data.target_calculator
            if not isinstance(target_calculator, LDOS) and not \
                    isinstance(target_calculator, DOS):
                raise Exception("Cannot calculate the band energy from this "
                                "observable.")
            target_calculator.\
                read_additional_calculation_data(
                self.data.get_snapshot_calculation_output(snapshot_number))

            target_calculator.read_from_array(actual_target)
            actual = target_calculator.band_energy

            target_calculator.read_from_array(predicted_target)
            predicted = target_calculator.band_energy
            return [actual, predicted,
                    target_calculator.band_energy_dft_calculation]

        elif observable == "number_of_electrons":
            target_calculator = self.data.target_calculator
            if not isinstance(target_calculator, LDOS) and not \
                    isinstance(target_calculator, DOS) and not \
                    isinstance(target_calculator, Density):
                raise Exception("Cannot calculate the band energy from this "
                                "observable.")
            target_calculator.\
                read_additional_calculation_data(
                self.data.get_snapshot_calculation_output(snapshot_number))

            actual = target_calculator.get_number_of_electrons(actual_target)

            predicted = target_calculator.get_number_of_electrons(predicted_target)
            return actual - predicted

        elif observable == "total_energy":
            target_calculator = self.data.target_calculator
            if not isinstance(target_calculator, LDOS):
                raise Exception("Cannot calculate the total energy from this "
                                "observable.")
            target_calculator.\
                read_additional_calculation_data(
                self.data.get_snapshot_calculation_output(snapshot_number))

            target_calculator.read_from_array(actual_target)
            actual = target_calculator.total_energy

            target_calculator.read_from_array(predicted_target)
            predicted = target_calculator.total_energy
            return actual - predicted

        elif observable == "total_energy_full":
            target_calculator = self.data.target_calculator
            if not isinstance(target_calculator, LDOS):
                raise Exception("Cannot calculate the total energy from this "
                                "observable.")
            target_calculator.\
                read_additional_calculation_data(
                self.data.get_snapshot_calculation_output(snapshot_number))

            target_calculator.read_from_array(actual_target)
            actual = target_calculator.total_energy

            target_calculator.read_from_array(predicted_target)
            predicted = target_calculator.total_energy
            return [actual, predicted,
                    target_calculator.total_energy_dft_calculation]

        elif observable == "density":
            target_calculator = self.data.target_calculator
            if not isinstance(target_calculator, LDOS) and \
                    not isinstance(target_calculator, Density):
                raise Exception("Cannot calculate the total energy from this "
                                "observable.")
            target_calculator.\
                read_additional_calculation_data(
                self.data.get_snapshot_calculation_output(snapshot_number))

            target_calculator.read_from_array(actual_target)
            actual = target_calculator.density

            target_calculator.read_from_array(predicted_target)
            predicted = target_calculator.density
            return np.mean(np.abs((actual - predicted) / actual)) * 100

        elif observable == "dos":
            target_calculator = self.data.target_calculator
            if not isinstance(target_calculator, LDOS) and \
                    not isinstance(target_calculator, DOS):
                raise Exception("Cannot calculate the total energy from this "
                                "observable.")
            target_calculator.\
                read_additional_calculation_data(
                self.data.get_snapshot_calculation_output(snapshot_number))

            target_calculator.read_from_array(actual_target)
            actual = target_calculator.density_of_states

            target_calculator.read_from_array(predicted_target)
            predicted = target_calculator.density_of_states
            return np.mean(np.abs((actual - predicted) / actual)) * 100




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

        optimal_batch_size = self.\
            _correct_batch_size_for_testing(grid_size,
                                            self.parameters.mini_batch_size)
        if optimal_batch_size != self.parameters.mini_batch_size:
            printout("Had to readjust batch size from",
                     self.parameters.mini_batch_size, "to",
                     optimal_batch_size, min_verbosity=0)
            self.parameters.mini_batch_size = optimal_batch_size
        self.number_of_batches_per_snapshot = int(grid_size /
                                                  self.parameters.
                                                  mini_batch_size)
