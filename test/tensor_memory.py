import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def test_tensor_memory(file):

    # Load an array as a numpy array
    loaded_array_raw = np.load(file)

    # Get dimensions of numpy array.
    dimension = np.shape(loaded_array_raw)
    datacount = dimension[0] * dimension[1] * dimension[2]

    # Reshape as datacount x featurelength
    loaded_array_raw = loaded_array_raw.astype(np.float32)
    loaded_array = loaded_array_raw.reshape([datacount, dimension[3]])

    # Check if reshaping allocated new memory.
    loaded_array_raw *= 10
    test1 = np.abs(np.sum(loaded_array) - np.sum(loaded_array_raw))
    if test1 > 0.000001:
        print("Test one: FAILURE. Reshaping changes memory allocation.")
    else:
        print("Test one: SUCCESS. Reshaping does not change memory allocation.")
    print("Diff is: ", test1)

    # simulate data splitting.
    index1 = int(80 / 100 * np.shape(loaded_array)[0])
    torch_tensor = torch.from_numpy(loaded_array[0:index1]).float()

    # Check if tensor and array are still the same.
    test1 = torch.abs(torch.sum(torch_tensor-loaded_array[0:index1]))
    if test1 > 0.000001:
        print("Test two: FAILURE. Tensor and array are different directly after reading and slicing.")
    else:
        print("Test two: SUCCESS. Tensor and array are the same directly after reading and slicing.")
    print("Diff is: ", test1)

    # Simulate data operation.
    loaded_array *= 10

    # Check if tensor and array are still the same.
    test1 = torch.abs(torch.sum(torch_tensor-loaded_array[0:index1]))
    if test1 > 0.000001:
        print("Test three: FAILURE. Tensor and array are different after data operation.")
    else:
        print("Test three: SUCCESS. Tensor and array are the same after data operation")
    print("Diff is: ", test1)

    # Simulate Tensor data handling in pytorch workflow.
    data_set = TensorDataset(torch_tensor, torch_tensor)
    data_loader = DataLoader(data_set, batch_size=index1)

    # Perform data operation again.
    loaded_array *= 10
    for (x, y) in data_loader:
        test1 = torch.abs(torch.sum(x - loaded_array[0:index1]))

    if test1 > 0.000001:
        print("Test four: FAILURE. Tensor and array are different after PyTorch operation.")
    else:
        print("Test four: SUCCESS. Tensor and array are the same after PyTorch operation")
    print("Diff is: ", test1)


if __name__ == "__main__":
    test_tensor_memory("../examples/data/Al_debug_2k_nr0.in.npy")

