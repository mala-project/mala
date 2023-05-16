"""Runner class for running networks."""
import os
from zipfile import ZipFile, ZIP_STORED

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np
import torch

from mala.common.parameters import ParametersRunning
from mala.network.network import Network
from mala.datahandling.data_scaler import DataScaler
from mala.datahandling.data_handler import DataHandler
from mala import Parameters


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

    def save_run(self, run_name, save_path="./", zip_run=True,
                 save_runner=False, additional_calculation_data=None):
        """
        Save the current run.

        Parameters
        ----------
        run_name : str
            Name under which the run should be saved.

        save_path : str
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
        model_file = run_name + ".network.pth"
        iscaler_file = run_name + ".iscaler.pkl"
        oscaler_file = run_name + ".oscaler.pkl"
        params_file = run_name + ".params.json"
        if save_runner:
            optimizer_file = run_name+".optimizer.pth"

        self.parameters_full.save(os.path.join(save_path, params_file))
        self.network.save_network(os.path.join(save_path, model_file))
        self.data.input_data_scaler.save(os.path.join(save_path, iscaler_file))
        self.data.output_data_scaler.save(os.path.join(save_path,
                                                       oscaler_file))

        files = [model_file, iscaler_file, oscaler_file, params_file]
        if save_runner:
            files += [optimizer_file]
        if zip_run:
            if additional_calculation_data is not None:
                additional_calculation_file = run_name+".info.json"
                if isinstance(additional_calculation_data, str):
                    self.data.target_calculator.\
                        read_additional_calculation_data(additional_calculation_data)
                    self.data.target_calculator.\
                        write_additional_calculation_data(os.path.join(save_path,
                                                          additional_calculation_file))
                elif isinstance(additional_calculation_data, bool):
                    if additional_calculation_data:
                        self.data.target_calculator. \
                            write_additional_calculation_data(os.path.join(save_path,
                                                              additional_calculation_file))

                files.append(additional_calculation_file)
            with ZipFile(os.path.join(save_path, run_name+".zip"), 'w',
                         compression=ZIP_STORED) as zip_obj:
                for file in files:
                    zip_obj.write(os.path.join(save_path, file), file)
                    os.remove(os.path.join(save_path, file))

    @classmethod
    def load_run(cls, run_name, path="./", zip_run=True,
                 params_format="json", load_runner=True,
                 prepare_data=False):
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
            loaded_params = run_name + ".params."+params_format
            loaded_info = run_name + ".info.json"

            zip_path = os.path.join(path, run_name + ".zip")
            with ZipFile(zip_path, 'r') as zip_obj:
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
            loaded_params = os.path.join(path, run_name +
                                         ".params."+params_format)

        loaded_params = Parameters.load_from_json(loaded_params)
        loaded_network = Network.load_from_file(loaded_params,
                                                loaded_network)
        loaded_iscaler = DataScaler.load_from_file(loaded_iscaler)
        loaded_oscaler = DataScaler.load_from_file(loaded_oscaler)
        new_datahandler = DataHandler(loaded_params,
                                      input_data_scaler=loaded_iscaler,
                                      output_data_scaler=loaded_oscaler,
                                      clear_data=(not prepare_data))
        if loaded_info is not None:
            new_datahandler.target_calculator.\
                read_additional_calculation_data(loaded_info,
                                                 data_type="json")

        if prepare_data:
            new_datahandler.prepare_data(reparametrize_scaler=False)

        if load_runner:
            if zip_run is True:
                with ZipFile(zip_path, 'r') as zip_obj:
                    loaded_runner = run_name + ".optimizer.pth"
                    if loaded_runner in zip_obj.namelist():
                        loaded_runner = zip_obj.open(loaded_runner)
            else:
                loaded_runner = os.path.join(run_name + ".optimizer.pth")

            loaded_runner = cls._load_from_run(loaded_params, loaded_network,
                                               new_datahandler,
                                               file=loaded_runner)
            return loaded_params, loaded_network, new_datahandler, \
                   loaded_runner
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
            return os.path.isfile(run_name+".zip")
        else:
            network_name = run_name + ".network.pth"
            iscaler_name = run_name + ".iscaler.pkl"
            oscaler_name = run_name + ".oscaler.pkl"
            param_name = run_name + ".params."+params_format
            return all(map(os.path.isfile, [iscaler_name, oscaler_name, param_name,
                                            network_name]))

    @classmethod
    def _load_from_run(cls, params, network, data, file=None):
        # Simply create a new runner. If the child classes have to implement
        # it theirselves.
        loaded_runner = cls(params, network, data)
        return loaded_runner

    def _forward_entire_snapshot(self, snapshot_number, data_set,
                                 data_set_type,
                                 number_of_batches_per_snapshot=0,
                                 batch_size=0):
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
        # Determine where the snapshot begins and ends.
        from_index = 0
        to_index = None

        for idx, snapshot in enumerate(self.data.parameters.
                                               snapshot_directories_list):
            if snapshot.snapshot_function == data_set_type:
                if idx == snapshot_number:
                    to_index = from_index + snapshot.grid_size
                    break
                else:
                    from_index += snapshot.grid_size
        grid_size = to_index-from_index

        if self.data.parameters.use_lazy_loading:
            data_set.return_outputs_directly = True
            actual_outputs = \
                (data_set
                 [from_index:to_index])[1]
        else:
            actual_outputs = \
                self.data.output_data_scaler.\
                inverse_transform(
                    (data_set[from_index:to_index])[1],
                    as_numpy=True)

        predicted_outputs = np.zeros((grid_size,
                                      self.data.output_dimension))

        for i in range(0, number_of_batches_per_snapshot):
            inputs, outputs = \
                data_set[from_index+(i * batch_size):from_index+((i + 1)
                                                                 * batch_size)]
            inputs = inputs.to(self.parameters._configuration["device"])
            predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                self.data.output_data_scaler.\
                inverse_transform(self.network(inputs).
                                  to('cpu'), as_numpy=True)

        # Restricting the actual quantities to physical meaningful values,
        # i.e. restricting the (L)DOS to positive values.
        predicted_outputs = self.data.target_calculator.\
            restrict_data(predicted_outputs)

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

        This includes e.g. horovod setup.
        """
        # See if we want to use horovod.
        if self.parameters_full.use_horovod:
            if self.parameters_full.use_gpu:
                # We cannot use "printout" here because this is supposed
                # to happen on every rank.
                if self.parameters_full.verbosity >= 2:
                    print("size=", hvd.size(), "global_rank=", hvd.rank(),
                          "local_rank=", hvd.local_rank(), "device=",
                          torch.cuda.get_device_name(hvd.local_rank()))
                # pin GPU to local rank
                torch.cuda.set_device(hvd.local_rank())
