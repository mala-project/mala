"""Tester class for testing a network."""
import glob
import os
from zipfile import ZipFile, ZIP_STORED

import ase.io
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np
import torch

from mala.common.parallelizer import printout, get_rank, barrier
from mala.network.predictor import Predictor
from mala.network.network import Network
from mala.datahandling.data_scaler import DataScaler
from mala.datahandling.data_handler import DataHandler
from mala import Parameters


class SimpleEnsemblePredictor(Predictor):
    """
    A class for running predictions using a neural network.

    It enables production-level inference.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Predictor object.

    networks : list
        List of networks used for predictions.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler, in this case not directly holding data, but serving
        as an interface to Target and Descriptor objects.
    """

    def __init__(self, params, networks, data):
        # copy the parameters into the class.
        super(SimpleEnsemblePredictor, self).__init__(params, networks[0], data)
        self.number_of_models = len(networks)
        self.networks = networks

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

        loaded_network : list
            All networks loaded for this predictor.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from file.

        new_runner : cls
            (Optional) The runner reconstructed from file. For Tester and
            Predictor class, this is just a newly instantiated object.
        """
        if zip_run is False:
            raise Exception("Multi model prediction only works with zipped"
                            " models.")

        if load_runner is False:
            raise Exception("Multi model prediction always requires Runner to"
                            " be loaded to manage multiple models.")

        runs = [os.path.basename(x) for x in
                glob.glob(os.path.join(path, run_name))]
        networks = []
        for idx, run in enumerate(runs):
            run = run.split(".zip")[0]
            loaded_network = run + ".network.pth"
            loaded_iscaler = run + ".iscaler.pkl"
            loaded_oscaler = run + ".oscaler.pkl"
            loaded_params = run + ".params."+params_format
            loaded_info = run + ".info.json"

            zip_path = os.path.join(path, run + ".zip")
            with ZipFile(zip_path, 'r') as zip_obj:
                loaded_params = zip_obj.open(loaded_params)
                loaded_network = zip_obj.open(loaded_network)
                loaded_iscaler = zip_obj.open(loaded_iscaler)
                loaded_oscaler = zip_obj.open(loaded_oscaler)
                if loaded_info in zip_obj.namelist():
                    loaded_info = zip_obj.open(loaded_info)
                else:
                    loaded_info = None

            loaded_params = Parameters.load_from_json(loaded_params)
            networks.append(Network.
                            load_from_file(loaded_params, loaded_network))
            if idx == 0:

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

            if idx == 0:
                with ZipFile(zip_path, 'r') as zip_obj:
                    loaded_runner = run + ".optimizer.pth"
                    if loaded_runner in zip_obj.namelist():
                        loaded_runner = zip_obj.open(loaded_runner)
                first_loaded_params = loaded_params
                first_loaded_data_handler = new_datahandler

        loaded_runner = cls._load_from_run(first_loaded_params,
                                           networks,
                                           first_loaded_data_handler,
                                           file=loaded_runner)
        return first_loaded_params, networks, first_loaded_data_handler, \
               loaded_runner

    def _forward_snap_descriptors(self, snap_descriptors,
                                  network,
                                  local_data_size=None):
        """Forward a scaled tensor of descriptors through the NN."""
        all_predicted_outputs = []
        for m in range(0, self.number_of_models):
            predicted_outputs = super(SimpleEnsemblePredictor, self).\
                _forward_snap_descriptors(snap_descriptors, self.networks[m],
                                          local_data_size=local_data_size)
            all_predicted_outputs.append(predicted_outputs)
        return all_predicted_outputs
