from itertools import product 
import torch 

class PatchesIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, snapshots_batch, patch_size):
        super(MyIterableDataset).__init__()
        self.snapshots_batch = snapshots_batch
        self.stopb = self.snapshots_batch.shape[0]
        self.stopx = self.snapshots_batch.shape[1] - patch_size + 1
        self.stopy = self.snapshots_batch.shape[2] - patch_size + 1
        self.stopz = self.snapshots_batch.shape[3] - patch_size + 1
        self.patch_size = patch_size 
        self.indices_iterator = product(range(self.stopb), 
                                        range(self.stopx), 
                                        range(self.stopy),
                                        range(self.stopz))

    def __iter__(self):
        return self
    
    def __next__(self):
            idx = next(self.indices_iterator)
            return self.snapshots_batch[idx[0],
                               idx[1] : idx[1] + self.patch_size,
                               idx[2] : idx[2] + self.patch_size,
                               idx[3] : idx[3] + self.patch_size,
                               :]

