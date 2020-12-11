'''Collection of normalization functions.'''

import torch


def standardization(data, mean, std):
    data = (data - mean) / std

def minmaxing(data, max, min):
    data = (data - min) / (max - min)

def normalize_three_tensors(tensor1, tensor2, tensor3, row_length, scale_standard, scale_max, use_row):
    """Normalizes three tensors at the same time. This is convenient for normalizing training, validation
    and test data at the same time."""

    # If we are not operating element-wise there is no need to recalculate
    # the "global values" all the time
    if use_row:
        if scale_standard:
            means = torch.mean([tensor1, tensor2, tensor3])
            stds = torch.std([tensor1, tensor2, tensor3])
        if scale_max:
            mins = torch.min([full_train_fp_np[row,:], validation_fp_np[row, :], test_fp_np[row, :]])
            maxs = torch.max([full_train_fp_np[row,:], validation_fp_np[row, :], test_fp_np[row, :]])
            for i in range(0,3):
                if (maxs[i] - mins[i] < 1e-12):
                    raise Exception("Error in normalization, Max and Min are too close.")

    for row in range(0, row_length):
        if scale_standard:
            if use_rows:
                means = torch.mean([tensor1[row,:], tensor2[row,:], tensor3[row,:]])
                stds = torch.std([tensor1[row,:], tensor2[row,:], tensor3[row,:]])
                standardization(tensor1[row,:], means[0], stds[0])
                standardization(tensor2[row,:], means[1], stds[1])
                standardization(tensor3[row,:], means[2], stds[2])
            else:
                standardization(tensor1, means[0], stds[0])
                standardization(tensor2, means[1], stds[1])
                standardization(tensor3, means[2], stds[2])
        if scale_max:
            if use_rows:
                mins = torch.min([tensor1[row,:], tensor2[row, :], tensor3[row, :]])
                maxs = torch.max([tensor1[row,:], tensor2[row, :], tensor3[row, :]])
                for i in range(0,3):
                    if (maxs[i] - mins[i] < 1e-12):
                        raise Exception("Error in normalization, Max and Min are too close.")
                minmaxing(tensor1[row,:], maxs[0], mins[0])
                minmaxing(tensor2[row,:], maxs[1], mins[1])
                minmaxing(tensor3[row,:], maxs[2], mins[2])
            else:
                minmaxing(tensor1, maxs[0], mins[0])
                minmaxing(tensor2, maxs[1], mins[1])
                minmaxing(tensor3, maxs[2], mins[2])
