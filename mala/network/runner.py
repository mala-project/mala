"""Runner class for running networks."""

import os
from zipfile import ZipFile, ZIP_STORED

from mala.common.parallelizer import printout

import numpy as np
import torch
import torch.distributed as dist

import mala
from mala.common.parallelizer import get_rank
from mala.common.parameters import ParametersRunning
from mala.datahandling.fast_tensor_dataset import FastTensorDataset
from mala.network.network import Network
from mala.datahandling.data_scaler import DataScaler
from mala.datahandling.data_handler import DataHandler
from mala import Parameters
from mala.targets.ldos import LDOS
from mala.targets.dos import DOS
from mala.targets.density import Density

from tqdm.auto import tqdm, trange


class Runner:
    """
    Parent class for all classes that in some sense "run" the network.

    That can be training, benchmarking, inference, etc.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Runner object.

    network : mala.network.network.Network
        Network which is being run.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the run.
    """

    def __init__(self, params, network, data, runner_dict=None):
        self.parameters_full: Parameters = params
        self.parameters: ParametersRunning = params.running
        self.network = network
        self.data = data
        self.__prepare_to_run()

    def _calculate_errors(
        self, actual_outputs, predicted_outputs, metrics, snapshot_number
    ):
        """
        Calculate the errors between the actual and predicted outputs.

        Parameters
        ----------
        actual_outputs : numpy.ndarray
            Actual outputs.

        predicted_outputs : numpy.ndarray
            Predicted outputs.

        metrics : list
            List of metrics to calculate.

        snapshot_number : int
            Snapshot number for which the errors are calculated.

        Returns
        -------
        errors : dict
            Dictionary containing the errors.
        """
        energy_metrics = [metric for metric in metrics if "energy" in metric]
        non_energy_metrics = [
            metric for metric in metrics if "energy" not in metric
        ]
        if len(energy_metrics) > 0:
            errors = self._calculate_energy_errors(
                actual_outputs,
                predicted_outputs,
                energy_metrics,
                snapshot_number,
            )
        else:
            errors = {}
        for metric in non_energy_metrics:
            try:
                if metric == "ldos":
                    error = np.mean((predicted_outputs - actual_outputs) ** 2)
                    errors[metric] = error

                elif metric == "density":
                    target_calculator = self.data.target_calculator
                    if not isinstance(
                        target_calculator, LDOS
                    ) and not isinstance(target_calculator, Density):
                        raise Exception(
                            "Cannot calculate density from this " "observable."
                        )
                    target_calculator.read_additional_calculation_data(
                        self.data.get_snapshot_calculation_output(
                            snapshot_number
                        )
                    )

                    target_calculator.read_from_array(actual_outputs)
                    actual = target_calculator.density

                    target_calculator.read_from_array(predicted_outputs)
                    predicted = target_calculator.density
                    errors[metric] = np.mean(np.abs(actual - predicted))

                elif metric == "density_relative":
                    target_calculator = self.data.target_calculator
                    if not isinstance(
                        target_calculator, LDOS
                    ) and not isinstance(target_calculator, Density):
                        raise Exception(
                            "Cannot calculate the density from this "
                            "observable."
                        )
                    target_calculator.read_additional_calculation_data(
                        self.data.get_snapshot_calculation_output(
                            snapshot_number
                        )
                    )

                    target_calculator.read_from_array(actual_outputs)
                    actual = target_calculator.density

                    target_calculator.read_from_array(predicted_outputs)
                    predicted = target_calculator.density
                    errors[metric] = (
                        np.mean(np.abs((actual - predicted) / actual)) * 100
                    )

                elif metric == "dos":
                    target_calculator = self.data.target_calculator
                    if not isinstance(
                        target_calculator, LDOS
                    ) and not isinstance(target_calculator, DOS):
                        raise Exception(
                            "Cannot calculate the DOS from this " "observable."
                        )
                    target_calculator.read_additional_calculation_data(
                        self.data.get_snapshot_calculation_output(
                            snapshot_number
                        )
                    )

                    target_calculator.read_from_array(actual_outputs)
                    actual = target_calculator.density_of_states

                    target_calculator.read_from_array(predicted_outputs)
                    predicted = target_calculator.density_of_states

                    errors[metric] = np.abs(actual - predicted).mean()

                elif metric == "dos_relative":
                    target_calculator = self.data.target_calculator
                    if not isinstance(
                        target_calculator, LDOS
                    ) and not isinstance(target_calculator, DOS):
                        raise Exception(
                            "Cannot calculate the relative DOS from this "
                            "observable."
                        )
                    target_calculator.read_additional_calculation_data(
                        self.data.get_snapshot_calculation_output(
                            snapshot_number
                        )
                    )

                    # We shift both the actual and predicted DOS by 1.0 to overcome
                    # numerical issues with the DOS having values equal to zero.
                    target_calculator.read_from_array(actual_outputs)
                    actual = target_calculator.density_of_states + 1.0

                    target_calculator.read_from_array(predicted_outputs)
                    predicted = target_calculator.density_of_states + 1.0

                    errors[metric] = (
                        np.ma.masked_invalid(
                            np.abs(
                                (actual - predicted)
                                / (np.abs(actual) + np.abs(predicted))
                            )
                        ).mean()
                        * 100
                    )
                else:
                    raise Exception(f"Invalid metric ({metric}) requested.")
            except ValueError as e:
                printout(
                    f"Error calculating observable: {metric} for snapshot {snapshot_number}",
                    min_verbosity=0,
                )
                printout(e, min_verbosity=2)
                errors[metric] = float("inf")
        return errors

    def _calculate_energy_errors(
        self, actual_outputs, predicted_outputs, energy_types, snapshot_number
    ):
        """
        Calculate the errors between the actual and predicted outputs.

        Parameters
        ----------
        actual_outputs : numpy.ndarray
            Actual outputs.

        predicted_outputs : numpy.ndarray
            Predicted outputs.

        energy_types : list
            List of energy types to calculate errors.

        snapshot_number : int
            Snapshot number for which the errors are calculated.
        """
        target_calculator = self.data.target_calculator
        output_file = self.data.get_snapshot_calculation_output(
            snapshot_number
        )
        if not output_file:
            raise Exception(
                "Output file needed for energy error calculations."
            )
        target_calculator.read_additional_calculation_data(output_file)

        errors = {}
        fe_actual = None
        fe_predicted = None
        try:
            fe_actual = target_calculator.get_self_consistent_fermi_energy(
                actual_outputs
            )
        except ValueError:
            errors = {
                energy_type: float("inf") for energy_type in energy_types
            }
            printout(
                "CAUTION! LDOS ground truth is so wrong that the "
                "estimation of the self consistent Fermi energy fails."
            )
            return errors
        try:
            fe_predicted = target_calculator.get_self_consistent_fermi_energy(
                predicted_outputs
            )
        except ValueError:
            errors = {
                energy_type: float("inf") for energy_type in energy_types
            }
            printout(
                "CAUTION! LDOS prediction is so wrong that the "
                "estimation of the self consistent Fermi energy fails."
            )
            return errors
        for energy_type in energy_types:
            if energy_type == "fermi_energy":
                fe_error = fe_predicted - fe_actual
                errors[energy_type] = fe_error
            elif energy_type == "band_energy":
                if not isinstance(target_calculator, LDOS) and not isinstance(
                    target_calculator, DOS
                ):
                    raise Exception(
                        "Cannot calculate the band energy from this observable."
                    )
                try:
                    target_calculator.read_from_array(actual_outputs)
                    be_actual = target_calculator.get_band_energy(
                        fermi_energy=fe_actual
                    )
                    target_calculator.read_from_array(predicted_outputs)
                    be_predicted = target_calculator.get_band_energy(
                        fermi_energy=fe_predicted
                    )
                    be_error = (be_predicted - be_actual) * (
                        1000 / len(target_calculator.atoms)
                    )
                    errors[energy_type] = be_error
                except ValueError:
                    errors[energy_type] = float("inf")
            elif energy_type == "band_energy_actual_fe":
                if not isinstance(target_calculator, LDOS) and not isinstance(
                    target_calculator, DOS
                ):
                    raise Exception(
                        "Cannot calculate the band energy from this observable."
                    )
                try:
                    target_calculator.read_from_array(predicted_outputs)
                    be_predicted_actual_fe = target_calculator.get_band_energy(
                        fermi_energy=fe_actual
                    )
                    be_error_actual_fe = (
                        be_predicted_actual_fe - be_actual
                    ) * (1000 / len(target_calculator.atoms))
                    errors[energy_type] = be_error_actual_fe
                except ValueError:
                    errors[energy_type] = float("inf")
            elif energy_type == "total_energy":
                if not isinstance(target_calculator, LDOS):
                    raise Exception(
                        "Cannot calculate the total energy from this "
                        "observable."
                    )
                try:
                    target_calculator.read_additional_calculation_data(
                        self.data.get_snapshot_calculation_output(
                            snapshot_number
                        )
                    )
                    target_calculator.read_from_array(actual_outputs)
                    te_actual = target_calculator.get_total_energy(
                        fermi_energy=fe_actual
                    )
                    target_calculator.read_from_array(predicted_outputs)
                    te_predicted = target_calculator.get_total_energy(
                        fermi_energy=fe_predicted
                    )
                    te_error = (te_predicted - te_actual) * (
                        1000 / len(target_calculator.atoms)
                    )
                    errors[energy_type] = te_error
                except ValueError:
                    errors[energy_type] = float("inf")
            elif energy_type == "total_energy_actual_fe":
                if not isinstance(target_calculator, LDOS):
                    raise Exception(
                        "Cannot calculate the total energy from this "
                        "observable."
                    )
                try:
                    target_calculator.read_from_array(predicted_outputs)
                    te_predicted_actual_fe = (
                        target_calculator.get_total_energy(
                            fermi_energy=fe_actual
                        )
                    )
                    te_error_actual_fe = (
                        te_predicted_actual_fe - te_actual
                    ) * (1000 / len(target_calculator.atoms))
                    errors[energy_type] = te_error_actual_fe
                except ValueError:
                    errors[energy_type] = float("inf")
            else:
                raise Exception(
                    f"Invalid energy type ({energy_type}) requested."
                )
        return errors

    def _calculate_energy_targets_and_predictions(
        self, actual_outputs, predicted_outputs, energy_types, snapshot_number
    ):
        """
        Calculate the energies corresponding to actual and predicted outputs.

        Parameters
        ----------
        actual_outputs : numpy.ndarray
            Actual outputs.

        predicted_outputs : numpy.ndarray
            Predicted outputs.

        energy_types : list
            List of energy types to calculate.

        snapshot_number : int
            Snapshot number for which the energies are calculated.
        """
        target_calculator = self.data.target_calculator
        output_file = self.data.get_snapshot_calculation_output(
            snapshot_number
        )
        if not output_file:
            raise Exception("Output file needed for energy calculations.")
        target_calculator.read_additional_calculation_data(output_file)

        targets = {}
        predictions = {}
        fe_actual = None
        fe_predicted = None
        try:
            fe_actual = target_calculator.get_self_consistent_fermi_energy(
                actual_outputs
            )
        except ValueError:
            targets = {energy_type: np.nan for energy_type in energy_types}
            predictions = {energy_type: np.nan for energy_type in energy_types}
            printout(
                "CAUTION! LDOS ground truth is so wrong that the "
                "estimation of the self consistent Fermi energy fails."
            )
            return targets, predictions
        try:
            fe_predicted = target_calculator.get_self_consistent_fermi_energy(
                predicted_outputs
            )
        except ValueError:
            targets = {energy_type: np.nan for energy_type in energy_types}
            predictions = {energy_type: np.nan for energy_type in energy_types}
            printout(
                "CAUTION! LDOS prediction is so wrong that the "
                "estimation of the self consistent Fermi energy fails."
            )
            return targets, predictions
        for energy_type in energy_types:
            if energy_type == "fermi_energy":
                targets[energy_type] = fe_actual
                predictions[energy_type] = fe_predicted
            elif energy_type == "band_energy":
                if not isinstance(target_calculator, LDOS) and not isinstance(
                    target_calculator, DOS
                ):
                    raise Exception(
                        "Cannot calculate the band energy from this observable."
                    )
                try:
                    target_calculator.read_from_array(actual_outputs)
                    be_actual = target_calculator.get_band_energy(
                        fermi_energy=fe_actual
                    )
                    target_calculator.read_from_array(predicted_outputs)
                    be_predicted = target_calculator.get_band_energy(
                        fermi_energy=fe_predicted
                    )
                    targets[energy_type] = (
                        be_actual * 1000 / len(target_calculator.atoms)
                    )
                    predictions[energy_type] = (
                        be_predicted * 1000 / len(target_calculator.atoms)
                    )
                except ValueError:
                    targets[energy_type] = np.nan
                    predictions[energy_type] = np.nan
            elif energy_type == "band_energy_actual_fe":
                if not isinstance(target_calculator, LDOS) and not isinstance(
                    target_calculator, DOS
                ):
                    raise Exception(
                        "Cannot calculate the band energy from this observable."
                    )
                try:
                    target_calculator.read_from_array(predicted_outputs)
                    be_predicted_actual_fe = target_calculator.get_band_energy(
                        fermi_energy=fe_actual
                    )
                    targets[energy_type] = (
                        be_actual * 1000 / len(target_calculator.atoms)
                    )
                    predictions[energy_type] = (
                        be_predicted_actual_fe
                        * 1000
                        / len(target_calculator.atoms)
                    )
                except ValueError:
                    targets[energy_type] = np.nan
                    predictions[energy_type] = np.nan
            elif energy_type == "total_energy":
                if not isinstance(target_calculator, LDOS):
                    raise Exception(
                        "Cannot calculate the total energy from this "
                        "observable."
                    )
                try:
                    target_calculator.read_additional_calculation_data(
                        self.data.get_snapshot_calculation_output(
                            snapshot_number
                        )
                    )
                    target_calculator.read_from_array(actual_outputs)
                    te_actual = target_calculator.get_total_energy(
                        fermi_energy=fe_actual
                    )
                    target_calculator.read_from_array(predicted_outputs)
                    te_predicted = target_calculator.get_total_energy(
                        fermi_energy=fe_predicted
                    )
                    targets[energy_type] = (
                        te_actual * 1000 / len(target_calculator.atoms)
                    )
                    predictions[energy_type] = (
                        te_predicted * 1000 / len(target_calculator.atoms)
                    )
                except ValueError:
                    targets[energy_type] = np.nan
                    predictions[energy_type] = np.nan
            elif energy_type == "total_energy_actual_fe":
                if not isinstance(target_calculator, LDOS):
                    raise Exception(
                        "Cannot calculate the total energy from this "
                        "observable."
                    )
                try:
                    target_calculator.read_from_array(predicted_outputs)
                    te_predicted_actual_fe = (
                        target_calculator.get_total_energy(
                            fermi_energy=fe_actual
                        )
                    )

                    targets[energy_type] = (
                        te_actual * 1000 / len(target_calculator.atoms)
                    )
                    predictions[energy_type] = (
                        te_predicted_actual_fe
                        * 1000
                        / len(target_calculator.atoms)
                    )
                except ValueError:
                    targets[energy_type] = np.nan
                    predictions[energy_type] = np.nan
            else:
                raise Exception(
                    f"Invalid energy type ({energy_type}) requested."
                )
        return targets, predictions

    def save_run(
        self,
        run_name,
        path="./",
        zip_run=True,
        save_runner=False,
        additional_calculation_data=None,
    ):
        """
        Save the current run.

        Parameters
        ----------
        run_name : str
            Name under which the run should be saved.

        path : str
            Path where to which the run.

        zip_run : bool
            If True, the entire run will be saved as a .zip file. If False,
            then the model will be saved as separate files.

        save_runner : bool
            If True, the Runner object will also be saved as object. This
            is unnecessary for most use cases, but used internally when
            a checkpoint is created during model training.

        additional_calculation_data : string or bool
            If this variable is a string, then additional calculation data will
            be copied from the file this variable points to and included in
            the saved model. If a a bool (and True), additional calculation
            data will be saved directly from the DataHandler object.
            Only has an effect in the .zip mode. If additional calculation
            data is already present in the DataHandler object, it can be saved
            by setting.
        """
        # If a model is trained via DDP, we need to make sure saving is only
        # performed on rank 0.
        if get_rank() == 0:
            model_file = run_name + ".network.pth"
            iscaler_file = run_name + ".iscaler.pkl"
            oscaler_file = run_name + ".oscaler.pkl"
            params_file = run_name + ".params.json"
            if save_runner:
                optimizer_file = run_name + ".optimizer.pth"
            os.makedirs(path, exist_ok=True)
            self.parameters_full.save(os.path.join(path, params_file))
            if self.parameters_full.use_ddp:
                self.network.module.save_network(
                    os.path.join(path, model_file)
                )
            else:
                self.network.save_network(os.path.join(path, model_file))
            self.data.input_data_scaler.save(os.path.join(path, iscaler_file))
            self.data.output_data_scaler.save(os.path.join(path, oscaler_file))

            files = [model_file, iscaler_file, oscaler_file, params_file]
            if save_runner:
                files += [optimizer_file]
            if zip_run:
                if additional_calculation_data is not None:
                    additional_calculation_file = run_name + ".info.json"
                    if isinstance(additional_calculation_data, str):
                        self.data.target_calculator.read_additional_calculation_data(
                            additional_calculation_data
                        )
                        self.data.target_calculator.write_additional_calculation_data(
                            os.path.join(path, additional_calculation_file)
                        )
                    elif isinstance(additional_calculation_data, bool):
                        if additional_calculation_data:
                            self.data.target_calculator.write_additional_calculation_data(
                                os.path.join(path, additional_calculation_file)
                            )

                    files.append(additional_calculation_file)
                with ZipFile(
                    os.path.join(path, run_name + ".zip"),
                    "w",
                    compression=ZIP_STORED,
                ) as zip_obj:
                    for file in files:
                        zip_obj.write(os.path.join(path, file), file)
                        os.remove(os.path.join(path, file))

    @classmethod
    def load_run(
        cls,
        run_name,
        path="./",
        zip_run=True,
        params_format="json",
        load_runner=True,
        prepare_data=False,
        load_with_mpi=None,
        load_with_gpu=None,
        load_with_ddp=None,
    ):
        """
        Load a run.

        Parameters
        ----------
        run_name : str
            Name under which the run is saved.

        path : str
            Path where the run is saved.

        zip_run : bool
            If True, MALA will attempt to load from a .zip file. If False,
            then separate files will be attempted to be loaded.

        params_format : str
            Can be "json" or "pkl", depending on what was saved by the model.
            Default is "json".

        load_runner : bool
            If True, a Runner object will be created/loaded for further use.

        prepare_data : bool
            If True, the data will be loaded into memory. This is needed when
            continuing a model training.

        load_with_mpi : bool or None
            Can be used to actively enable/disable MPI during loading.
            Default is None, so that the MPI parameters set during
            training/saving of the model are not overwritten.
            If MPI is to be used in concert with GPU during training,
            MPI already has to be activated here, if it was not activated
            during training!

        load_with_gpu : bool or None
            Can be used to actively enable/disable GPU during loading.
            Default is None, so that the GPU parameters set during
            training/saving of the model are not overwritten.
            If MPI is to be used in concert with GPU during training,
            it is advised that GPU usage is activated here, if it was not
            activated during training. Can also be used to activate a CPU
            based inference, by setting it to False.

        load_with_ddp : bool or None
            Can be used to actively disable DDP (pytorch distributed
            data parallel used for parallel training) during loading.
            Default is None, which for loading a Trainer object will not
            interfere with DDP settings. For Predictor and Tester class,
            this command will automatically disable DDP during loading,
            as inference is using MPI rather than DDP for parallelization.

        Return
        ------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved to file.

        loaded_network : mala.network.network.Network
            The network saved to file.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from file.

        new_runner : cls
            (Optional) The runner reconstructed from file. For Tester and
            Predictor class, this is just a newly instantiated object.
        """
        loaded_info = None
        if zip_run is True:
            loaded_network = run_name + ".network.pth"
            loaded_iscaler = run_name + ".iscaler.pkl"
            loaded_oscaler = run_name + ".oscaler.pkl"
            loaded_params = run_name + ".params." + params_format
            loaded_info = run_name + ".info.json"

            zip_path = os.path.join(path, run_name + ".zip")
            with ZipFile(zip_path, "r") as zip_obj:
                loaded_params = zip_obj.open(loaded_params)
                loaded_network = zip_obj.open(loaded_network)
                loaded_iscaler = zip_obj.open(loaded_iscaler)
                loaded_oscaler = zip_obj.open(loaded_oscaler)
                if loaded_info in zip_obj.namelist():
                    loaded_info = zip_obj.open(loaded_info)
                else:
                    loaded_info = None

        else:
            loaded_network = os.path.join(path, run_name + ".network.pth")
            loaded_iscaler = os.path.join(path, run_name + ".iscaler.pkl")
            loaded_oscaler = os.path.join(path, run_name + ".oscaler.pkl")
            loaded_params = os.path.join(
                path, run_name + ".params." + params_format
            )

        # Neither Predictor nor Runner classes can work with DDP.
        if cls is mala.Trainer:
            loaded_params = Parameters.load_from_json(loaded_params)
        else:
            loaded_params = Parameters.load_from_json(
                loaded_params, force_no_ddp=True
            )

        # MPI has to be specified upon loading, in contrast to GPU.
        if load_with_mpi is not None:
            loaded_params.use_mpi = load_with_mpi
        if load_with_gpu is not None:
            loaded_params.use_gpu = load_with_gpu

        loaded_network = Network.load_from_file(loaded_params, loaded_network)
        loaded_iscaler = DataScaler.load_from_file(loaded_iscaler)
        loaded_oscaler = DataScaler.load_from_file(loaded_oscaler)
        new_datahandler = DataHandler(
            loaded_params,
            input_data_scaler=loaded_iscaler,
            output_data_scaler=loaded_oscaler,
            clear_data=(not prepare_data),
        )
        if loaded_info is not None:
            new_datahandler.target_calculator.read_additional_calculation_data(
                loaded_info, data_type="json"
            )

        if prepare_data:
            new_datahandler.prepare_data(reparametrize_scaler=False)

        if load_runner:
            if zip_run is True:
                with ZipFile(zip_path, "r") as zip_obj:
                    loaded_runner = run_name + ".optimizer.pth"
                    if loaded_runner in zip_obj.namelist():
                        loaded_runner = zip_obj.open(loaded_runner)
            else:
                loaded_runner = os.path.join(run_name + ".optimizer.pth")

            loaded_runner = cls._load_from_run(
                loaded_params,
                loaded_network,
                new_datahandler,
                file=loaded_runner,
            )
            return (
                loaded_params,
                loaded_network,
                new_datahandler,
                loaded_runner,
            )
        else:
            return loaded_params, loaded_network, new_datahandler

    @classmethod
    def run_exists(cls, run_name, params_format="json", zip_run=True):
        """
        Check if a run exists.

        Parameters
        ----------
        run_name : str
            Name under which the run is saved.

        zip_run : bool
            If True, MALA will look for a .zip file. If False,
            then separate files will be attempted to be loaded.

        params_format : str
            Can be "json" or "pkl", depending on what was saved by the model.
            Default is "json".

        Returns
        -------
        exists : bool
            If True, the model exists.
        """
        if zip_run is True:
            return os.path.isfile(run_name + ".zip")
        else:
            network_name = run_name + ".network.pth"
            iscaler_name = run_name + ".iscaler.pkl"
            oscaler_name = run_name + ".oscaler.pkl"
            param_name = run_name + ".params." + params_format
            return all(
                map(
                    os.path.isfile,
                    [iscaler_name, oscaler_name, param_name, network_name],
                )
            )

    @classmethod
    def _load_from_run(cls, params, network, data, file=None):
        # Simply create a new runner. If the child classes have to implement
        # it theirselves.
        loaded_runner = cls(params, network, data)
        return loaded_runner

    def _forward_entire_snapshot(
        self,
        snapshot_number,
        data_set,
        data_set_type,
        number_of_batches_per_snapshot=0,
        batch_size=0,
    ):
        """
        Forward a snapshot through the network, get actual/predicted output.

        Parameters
        ----------
        snapshot_number : int
            Snapshot for which the prediction is done.
            GLOBAL snapshot number, i.e. across the entire list.

        number_of_batches_per_snapshot : int
            Number of batches that lie within a snapshot.

        batch_size : int
            Batch size used for forward pass.

        Returns
        -------
        actual_outputs : numpy.ndarray
            Actual outputs for snapshot.

        predicted_outputs : numpy.ndarray
            Precicted outputs for snapshot.
        """
        # Ensure the Network is on the correct device.
        # This line is necessary because GPU acceleration may have been
        # activated AFTER loading a model.
        self.network.to(self.network.params._configuration["device"])

        # Determine where the snapshot begins and ends.
        from_index = 0
        to_index = None

        for idx, snapshot in enumerate(
            self.data.parameters.snapshot_directories_list
        ):
            if snapshot.snapshot_function == data_set_type:
                if idx == snapshot_number:
                    to_index = from_index + snapshot.grid_size
                    break
                else:
                    from_index += snapshot.grid_size
        grid_size = to_index - from_index

        if isinstance(data_set, FastTensorDataset):
            predicted_outputs = np.zeros(
                (grid_size, self.data.output_dimension)
            )
            actual_outputs = np.zeros((grid_size, self.data.output_dimension))

            for i in range(len(data_set)):
                inputs, outputs = data_set[from_index + i]
                inputs = inputs.to(self.parameters._configuration["device"])
                predicted_outputs[
                    i * data_set.batch_size : (i + 1) * data_set.batch_size, :
                ] = self.data.output_data_scaler.inverse_transform(
                    self.network(inputs).to("cpu"), as_numpy=True
                )
                actual_outputs[
                    i * data_set.batch_size : (i + 1) * data_set.batch_size, :
                ] = self.data.output_data_scaler.inverse_transform(
                    torch.tensor(outputs), as_numpy=True
                )
        else:
            if self.data.parameters.use_lazy_loading:
                data_set.return_outputs_directly = True
                actual_outputs = (data_set[from_index:to_index])[1]
            else:
                actual_outputs = (
                    self.data.output_data_scaler.inverse_transform(
                        (data_set[from_index:to_index])[1], as_numpy=True
                    )
                )

            predicted_outputs = np.zeros(
                (grid_size, self.data.output_dimension)
            )
            for i in range(0, number_of_batches_per_snapshot):
                inputs, outputs = data_set[
                    from_index
                    + (i * batch_size) : from_index
                    + ((i + 1) * batch_size)
                ]
                inputs = inputs.to(self.parameters._configuration["device"])
                predicted_outputs[i * batch_size : (i + 1) * batch_size, :] = (
                    self.data.output_data_scaler.inverse_transform(
                        self.network(inputs).to("cpu"), as_numpy=True
                    )
                )

        # Restricting the actual quantities to physical meaningful values,
        # i.e. restricting the (L)DOS to positive values.
        predicted_outputs = self.data.target_calculator.restrict_data(
            predicted_outputs
        )

        # It could be that other operations will be happening with the data
        # set, so it's best to reset it.
        if self.data.parameters.use_lazy_loading:
            data_set.return_outputs_directly = False

        return actual_outputs, predicted_outputs

    @staticmethod
    def _correct_batch_size_for_testing(datasize, batchsize):
        """
        Get the correct batch size for testing.

        For testing snapshot the batch size needs to be such that
        data_per_snapshot / batch_size will result in an integer division
        without any residual value.
        """
        new_batch_size = batchsize
        if datasize % new_batch_size != 0:
            while datasize % new_batch_size != 0:
                new_batch_size += 1
        return new_batch_size

    def __prepare_to_run(self):
        """
        Prepare the Runner to run the Network.

        This includes e.g. ddp setup.
        """
        # See if we want to use ddp.
        if self.parameters_full.use_ddp:
            if self.parameters_full.use_gpu:
                # We cannot use "printout" here because this is supposed
                # to happen on every rank.
                size = dist.get_world_size()
                rank = dist.get_rank()
                local_rank = int(os.environ.get("LOCAL_RANK"))
                if self.parameters_full.verbosity >= 2:
                    print(
                        "size=",
                        size,
                        "global_rank=",
                        rank,
                        "local_rank=",
                        local_rank,
                        "device=",
                        torch.cuda.get_device_name(local_rank),
                    )
                # pin GPU to local rank
                torch.cuda.set_device(local_rank)
