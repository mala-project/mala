"""DataSet for lazy-loading."""
import os

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class.
    pass

import numpy as np
import torch
from torch.utils.data import Dataset

from mala.datahandling.snapshot import Snapshot
from mala.common.parallelizer import printout, barrier


class LazyLoadDatasetClustered(torch.utils.data.Dataset):
    """
    DataSet class for lazy loading.

    Only loads snapshots in the memory that are currently being processed.
    Uses a "caching" approach of keeping the last used snapshot in memory,
    until values from a new ones are used. Therefore, shuffling at DataSampler
    / DataLoader level is discouraged to the point that it was disabled.
    Instead, we mix the snapshot load order here ot have some sort of mixing
    at all.

    Parameters
    ----------
    input_dimension : int
        Dimension of an input vector.

    output_dimension : int
        Dimension of an output vector.

    input_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the input data.

    output_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the output data.

    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        Used to do unit conversion on input data.

    target_calculator : mala.targets.target.Target or derivative
        Used to do unit conversion on output data.

    grid_dimensions : list
        Dimensions of the grid (x,y,z).

    grid_size : int
        Size of the grid (x*y*z), i.e. the number of datapoints per
        snapshot.

    use_horovod : bool
        If true, it is assumed that horovod is used.

    input_requires_grad : bool
        If True, then the gradient is stored for the inputs.


    """

    def __init__(self, input_dimension, output_dimension, input_data_scaler,
                 output_data_scaler, descriptor_calculator,
                 target_calculator, grid_dimensions, grid_size, use_horovod,
                 number_of_clusters, train_ratio, sample_ratio,
                 input_requires_grad=False):
        self.snapshot_list = []
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_data_scaler = input_data_scaler
        self.output_data_scaler = output_data_scaler
        self.descriptor_calculator = descriptor_calculator
        self.target_calculator = target_calculator
        self.grid_dimensions = grid_dimensions
        self.grid_size = grid_size
        self.number_of_snapshots = 0
        self.total_size = 0
        self.descriptors_contain_xyz = self.descriptor_calculator.\
            descriptors_contain_xyz
        self.currently_loaded_file = None
        self.input_data = np.empty(0)
        self.output_data = np.empty(0)
        self.use_horovod = use_horovod
        self.return_outputs_directly = False
        self.input_requires_grad = input_requires_grad

        # Clustering specific things.
        self.number_of_clusters = number_of_clusters
        self.train_ratio = train_ratio
        self.sample_ratio = sample_ratio
        self.clustered_inputs = np.zeros(1)
        self.samples_per_cluster = np.zeros(1)
        self.cluster_idxs = [None] * self.number_of_clusters
        self.sampling_idxs = [None] * self.number_of_clusters
        self.current_sampling_idx = [None] * self.number_of_clusters

    @property
    def return_outputs_directly(self):
        """
        Control whether outputs are actually transformed.

        Has to be False for training. In the testing case,
        Numerical errors are smaller if set to True.
        """
        return self._return_outputs_directly

    @return_outputs_directly.setter
    def return_outputs_directly(self, value):
        self._return_outputs_directly = value

    def add_snapshot_to_dataset(self, snapshot: Snapshot):
        """
        Add a snapshot to a DataSet.

        Afterwards, the DataSet can and will load this snapshot as needed.

        Parameters
        ----------
        snapshot : mala.datahandling.snapshot.Snapshot
            Snapshot that is to be added to this DataSet.

        """
        self.snapshot_list.append(snapshot)
        self.number_of_snapshots += 1
        self.total_size = int(self.number_of_snapshots*self.grid_size *
                              self.sample_ratio)

    def __cluster_snapshot(self, snapshot_idx):
        # Since pgkmeans is taking a while to load and is only rarely
        # needed, I'll make this import per-demand.
        import pqkmeans

        # Load the data into memory, and transform it as necessary.
        # I know, the here-and-there transform via torch is ugly, but
        # currently, the MALA scalers only support torch tensors.
        file_path = os.path.join(self.snapshot_list[snapshot_idx].
                                 input_npy_directory,
                                 self.snapshot_list[snapshot_idx].
                                 input_npy_file)
        input_data = np.load(file_path)
        # Transform the data.
        if self.descriptors_contain_xyz:
            input_data = input_data[:, :, :, 3:]
        input_data = input_data.reshape([self.grid_size, self.input_dimension])
        input_data *= \
            self.descriptor_calculator.\
            convert_units(1, self.snapshot_list[snapshot_idx].input_units)
        input_data = input_data.astype(np.float32)
        input_data = torch.from_numpy(input_data).float()
        input_data = self.input_data_scaler.transform(input_data)
        input_data = np.array(input_data)

        # Pad the vector to be a power of 2.
        power2 = int(np.ceil(np.log2(input_data.shape[1])))
        input_data_padded = np.zeros([input_data.shape[0], 2**power2])
        input_data_padded[:input_data.shape[0], :input_data.shape[1]] = \
            input_data
        np.random.shuffle(input_data_padded)

        # Determine the number of subdimensions and Ks. The latter is currently
        # fixed.
        nsbdm = int(2 ** np.floor(power2 / 2.0))
        Ks = 256
        printout("Clustering: Begin encoder training with subdimension", nsbdm,
                 "and", Ks, "k-s.")

        # Set up the encoder and fit it.
        encoder = pqkmeans.encoder.PQEncoder(num_subdim=nsbdm, Ks=Ks)
        fit_samples = int(self.grid_size*self.train_ratio)
        fit_mask = np.zeros([self.grid_size], dtype=bool)
        fit_mask[:fit_samples] = True
        fit_mask = np.random.permutation(fit_mask)
        encoder.fit(input_data_padded[fit_mask])

        # Use the encoder for transformation.
        printout("Clustering: Begin input transformation.")
        input_data_pqcode = encoder.transform(input_data_padded)

        del input_data
        del input_data_padded

        # Use the result of this transformation for the K-Means.
        printout("Clustering: Begin K-Means set.")
        kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder,
                                              k=self.number_of_clusters)

        # In turn use the result of this for prediction.
        printout("Begin Kmeans fit predict")
        clustered_inputs = kmeans.fit_predict(input_data_pqcode)

        return clustered_inputs

    def cluster_dataset(self):
        """
        Calculate clusters for dataset (individually per snapshot).

        .. important:: Only call this function AFTER all snapshots were added.
        """
        # Clustered inputs holds the cluster a snapshot belongs to for every
        # input of that snapshot.
        self.clustered_inputs = np.zeros([self.number_of_snapshots,
                                          self.grid_size])
        self.samples_per_cluster = np.zeros([self.number_of_snapshots,
                                             self.number_of_clusters])
        for idx in range(0, len(self.snapshot_list)):
            # For each snapshot, we calculate which cluster a particular
            # grid point would belong to.
            printout("Clustering file %d" % idx)
            self.clustered_inputs[idx, :] = \
                self.__cluster_snapshot(idx)

            # Count how many samples we have per cluster, per snapshot.
            for i in range(0, self.number_of_clusters):
                self.samples_per_cluster[idx, i] = \
                    np.sum(self.clustered_inputs[idx, :] == i, dtype=np.int64)

            if np.sum(self.samples_per_cluster[idx, :]) != self.grid_size:
                raise ValueError("Sum of clustered inputs is different from"
                                 " number of inputs itself.")

        barrier()

        # In the original code there is a mixing at this point. I omit this
        # here because we do it later in the training routine.

    def mix_datasets(self):
        """
        Mix the order of the snapshots.

        With this, there can be some variance between runs.
        """
        if self.number_of_snapshots > 1:
            used_perm = torch.randperm(self.number_of_snapshots)
            barrier()
            if self.use_horovod:
                used_perm = hvd.broadcast(used_perm, 0)

            # Not only the snapshots, but also the clustered inputs and samples
            # per clusters have to be permutated.
            self.snapshot_list = [self.snapshot_list[i] for i in used_perm]
            self.clustered_inputs = self.clustered_inputs[used_perm]
            self.samples_per_cluster = self.samples_per_cluster[used_perm]
            self.get_new_data(0)

    def get_new_data(self, file_index):
        """
        Read a new snapshot into RAM.

        Parameters
        ----------
        file_index : i
            File to be read.
        """
        # Load the data into RAM.
        self.input_data = \
            np.load(os.path.join(
                    self.snapshot_list[file_index].input_npy_directory,
                    self.snapshot_list[file_index].input_npy_file))
        self.output_data = \
            np.load(os.path.join(
                    self.snapshot_list[file_index].output_npy_directory,
                    self.snapshot_list[file_index].output_npy_file))

        # Transform the data.
        if self.descriptors_contain_xyz:
            self.input_data = self.input_data[:, :, :, 3:]
        self.input_data = \
            self.input_data.reshape([self.grid_size, self.input_dimension])
        self.input_data *= \
            self.descriptor_calculator.\
            convert_units(1, self.snapshot_list[file_index].input_units)
        self.input_data = self.input_data.astype(np.float32)
        self.input_data = torch.from_numpy(self.input_data).float()
        self.input_data = self.input_data_scaler.transform(self.input_data)
        self.input_data.requires_grad = self.input_requires_grad

        self.output_data = \
            self.output_data.reshape([self.grid_size, self.output_dimension])
        self.output_data *= \
            self.target_calculator.\
            convert_units(1, self.snapshot_list[file_index].output_units)
        if self.return_outputs_directly is False:
            self.output_data = np.array(self.output_data)
            self.output_data = self.output_data.astype(np.float32)
            self.output_data = torch.from_numpy(self.output_data).float()
            self.output_data = \
                self.output_data_scaler.transform(self.output_data)

        # Save which data we have currently loaded.
        self.currently_loaded_file = file_index

        # Reset the clustering indices.
        # cluster_idxs will hold all the indices belonging to a specific
        # cluster (per snapshot).
        # sampling_idx holds the indices from which to sample for a specific
        # snapshot.
        for i in range(self.number_of_clusters):
            self.cluster_idxs[i] = np.arange(self.grid_size)[self.clustered_inputs[self.currently_loaded_file, :] == i]
            self.sampling_idxs[i] = np.random.permutation(np.arange(self.samples_per_cluster[self.currently_loaded_file, i], dtype=np.int64))
            self.current_sampling_idx[i] = 0

    def __get_clustered_idx(self, idx, file_idx):
        cluster = int(idx % self.number_of_clusters)
        num_cluster_samples = \
            self.samples_per_cluster[self.currently_loaded_file, cluster]

        # Rejection sampling - if this file does not contain any instance of
        # this cluster at all, different cluster will be tried.
        if num_cluster_samples == 0:
            bad_iters = 0

            while num_cluster_samples == 0:
                cluster = np.random.randint(self.number_of_clusters)
                num_cluster_samples = self.samples_per_cluster[file_idx,
                                                               cluster]

                if bad_iters > 100:
                    raise ValueError("100 successive bad clusters. Exiting.")

                bad_iters += 1

        # If the cluster is alright, we can sample from it.
        front = self.sampling_idxs[cluster]
        back = self.current_sampling_idx[cluster] % num_cluster_samples
        back = int(back)
        idx_within_cluster = front[back]
        self.current_sampling_idx[cluster] += 1
        print(idx, file_idx, cluster,
              self.cluster_idxs[cluster][idx_within_cluster])
        return self.cluster_idxs[cluster][idx_within_cluster]

    def __getitem__(self, idx):
        """
        Get an item of the DataSet.

        Parameters
        ----------
        idx : int
            Requested index. NOTE: Slices are currently NOT supported.

        Returns
        -------
        inputs, outputs : torch.Tensor
            The requested inputs and outputs
        """
        # We sample self.grid_size*self.sample_ratio per file.
        file_index = idx // int(self.grid_size*self.sample_ratio)

        # Find out if new data is needed.
        if file_index != self.currently_loaded_file:
            self.get_new_data(file_index)

        sample_idx = self.__get_clustered_idx(idx, file_index)
        return self.input_data[sample_idx], \
            self.output_data[sample_idx]

    def __len__(self):
        """
        Get the length of the DataSet.

        Returns
        -------
        length : int
            Number of data points in DataSet.
        """
        return self.total_size
