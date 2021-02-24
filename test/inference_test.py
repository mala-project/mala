from fesl.common.parameters import Parameters
from fesl.datahandling.data_handler import DataHandler
from fesl.datahandling.data_scaler import DataScaler
from fesl.network.network import Network
from fesl.network.tester import Tester
from fesl.common.parameters import printout
import numpy as np
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al256_reduced/"
param_path = get_data_repo_path()+"example08_data/"
params_path = param_path+"ex08_params.pkl"
network_path = param_path+"ex08_network.pth"
input_scaler_path = param_path+"ex08_iscaler.pkl"
output_scaler_path = param_path+"ex08_oscaler.pkl"

def test_inference(accuracy = 1e-16, use_lazy_loading = False):
    # First we load Parameters and network.
    new_parameters = Parameters.load_from_file(params_path, no_snapshots=True)
    new_parameters.data.data_splitting_snapshots = ["te", "te", "te"]
    new_parameters.data.use_lazy_loading = use_lazy_loading
    new_parameters.running.mini_batch_size = 9
    new_parameters.running.mini_batch_size = 46
    new_parameters.running.mini_batch_size = 46

    new_network = Network.load_from_file(new_parameters, network_path)

    # We use a data handler object to read the data we want to investigate.
    # We need to make sure that the same scaling is used.
    iscaler = DataScaler.load_from_file(input_scaler_path)
    oscaler = DataScaler.load_from_file(output_scaler_path)
    inference_data_handler = DataHandler(new_parameters, input_data_scaler=iscaler, output_data_scaler=oscaler)
    inference_data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path, "Al_debug_2k_nr0.out.npy", data_path,
                              output_units="1/Ry")
    inference_data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path, "Al_debug_2k_nr1.out.npy", data_path,
                              output_units="1/Ry")
    inference_data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path, "Al_debug_2k_nr2.out.npy", data_path,
                              output_units="1/Ry")

    inference_data_handler.prepare_data()


    for n in [46,99,500,1977]:
        new_parameters.running.mini_batch_size = n
        tester = Tester(new_parameters)
        tester.set_data(new_network, inference_data_handler)
        for i in range(0,inference_data_handler.nr_snapshots):
            raw_outputs = inference_data_handler.target_calculator.convert_units(np.load(data_path+"Al_debug_2k_nr"+str(i)+".out.npy"), in_units="1/Ry")
            raw_predicted_outputs = np.load(data_path+"Al_debug_2k_nr"+str(i)+".in.npy")
            raw_predicted_outputs = inference_data_handler.raw_numpy_to_converted_scaled_tensor(raw_predicted_outputs, "in", None)
            raw_predicted_outputs = new_network.do_prediction(raw_predicted_outputs)
            raw_predicted_outputs = inference_data_handler.output_data_scaler.inverse_transform(raw_predicted_outputs, as_numpy=True)

            actual_ldos, predicted_ldos  = tester.test_snapshot(i)
            if np.abs(predicted_ldos.sum()-raw_predicted_outputs.sum()) > accuracy:
                printout("Accuracy not met for prediction with accuracy being {0}".format(np.abs(predicted_ldos.sum()-raw_predicted_outputs.sum())))
                printout("Parameters are batchsize: {0}, lazy loading: {1}".format(n, new_parameters.data.use_lazy_loading))
                return False
            if np.abs(actual_ldos.sum()-raw_outputs.sum()) > accuracy:
                printout("Accuracy not met for raw data with accuracy being {0}".format(np.abs(actual_ldos.sum()-raw_outputs.sum())))
                printout("Batchsize: {0}, lazy loading: {1}".format(n, new_parameters.data.use_lazy_loading))
                return False
    return True


def run_inference_test():
    # Without lazy loading, there is a bigger numerical error in the actual outputs.
    # This is due to the fact that we scale and rescale data on 32 byte float arrays; lazy loading is therefore
    # drastically advised.
    if test_inference(accuracy=1e-6, use_lazy_loading=False) is False:
        return False
    if test_inference(accuracy=1e-12, use_lazy_loading=True) is False:
        return False
    return True

if __name__ == "__main__":
    test1 = run_inference_test()
    printout("Inference test - success?:", test1)
