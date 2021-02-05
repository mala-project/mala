import torch
from torch.utils.data import Dataset



class LazyLoadDataset(torch.utils.data.Dataset):
    def __init__(self):
        test = 0

    def __getindex__(self, idx):
        return load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)
