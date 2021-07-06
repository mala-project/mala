#FP LDOS, Data Loaders

import os, sys
import numpy as np
import timeit
#import torch.nn as nn
#import torch.nn.functional as F

import torch
import torch.multiprocessing as mp

#import torch.utils.Dataset
import torch.utils.data.distributed
import torch.utils.data
import torch.utils
import horovod.torch as hvd

sys.path.append("./src/charm/clustering")

import cluster_fingerprints

# Big Data Dataset for training data that does not fit into memory
class Big_Charm_Clustered_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, \
                 input_fpaths, \
                 output_fpaths, \
                 num_samples, \
                 input_sample_shape, \
                 output_sample_shape, \
                 input_subset, \
                 output_subset, \
                 input_scaler_kwargs={}, \
                 output_scaler_kwargs={}): #, \
                 #do_reset=True):
        # input:
        ## args:                    Argparser args
        ## input_fpaths:            paths to input numpy files
        ## output_fpaths:           paths to output numpy files
        ## num_samples:             number of samples per file
        ## input_sample_shape:      shape of input sample
        ## output_sample_shape:     shape of output sample
        ## input_subset:            take subset of numpy file sample to 
        ##                          fit input_sample_shape
        ## output_subset:           take subset of numpy file sample to 
        ##                          fit input_sample_shape
        ## input_scaler_kwargs:     dict of input scaler options
        ## output_scalar_kwargs:    dict of output scaler options 
       
        self.args = args

        self.input_fpaths = input_fpaths
        self.output_fpaths = output_fpaths

        self.num_samples = num_samples

        self.input_shape = np.insert(input_sample_shape, 0, num_samples)
        self.output_shape = np.insert(output_sample_shape, 0, num_samples)

        self.input_mask = np.zeros(input_sample_shape, dtype=bool)
        self.output_mask = np.zeros(output_sample_shape, dtype=bool)

        self.input_mask[input_subset] = True
        self.output_mask[output_subset] = True

        self.num_files = len(input_fpaths)

        # Cluster params
        self.num_clusters = args.num_clusters
        self.cluster_train_ratio = args.cluster_train_ratio
        self.cluster_sample_ratio = args.cluster_sample_ratio


        self.reset = True
        if (self.num_files == 1):
            self.reset = False

        if (self.num_files == 0):
            raise ValueError("\n\nNo files provided to the Big Charm Dataset. Exiting.\n\n")
        if (self.num_files != len(output_fpaths)):
            raise ValueError("\nInput file list not equal in length " + \
                             "with Output file list. Exiting.\n\n")

        tic = timeit.default_timer()
        print("Input scaling.")
        self.input_scaler  = Big_Data_Scaler(input_fpaths, \
                                             num_samples, \
                                             input_sample_shape, \
                                             input_subset, \
                                             **input_scaler_kwargs)
        toc = timeit.default_timer()

        self.is_input_scaled = not self.input_scaler.no_scaling
        print("Input Scaler Timing: %4.4f" % (toc - tic))

        hvd.allreduce(torch.tensor(0), name="barrier")
        
        tic = timeit.default_timer()
        print("Output scaling.")
        self.output_scaler = Big_Data_Scaler(output_fpaths, \
                                             num_samples, \
                                             output_sample_shape, \
                                             output_subset, \
                                             **output_scaler_kwargs)
        toc = timeit.default_timer()                            

        self.is_output_scaled = not self.output_scaler.no_scaling
        print("Output Scaler Timing: %4.4f" % (toc - tic))
       
        hvd.allreduce(torch.tensor(0), name="barrier")
        
        if (hvd.rank() == 0):
            print("Input FP Factors")
            self.input_scaler.print_factors()
            print("Output LDOS Factors")
            self.output_scaler.print_factors()

        hvd.allreduce(torch.tensor(0), name="barrier")

        
        self.clustered_inputs = np.zeros([self.num_files, self.num_samples])

        self.samples_per_cluster = np.zeros([self.num_files, args.num_clusters])

        for idx, fpath in enumerate(input_fpaths):
            print("Clustering file %d: %s" % (idx, fpath))
            
            tic = timeit.default_timer()
            self.clustered_inputs[idx, :] = cluster_fingerprints.cluster_snapshot(fpath, \
                                                                                  self.num_samples, \
                                                                                  self.input_shape, \
                                                                                  self.input_mask, \
                                                                                  self.input_scaler, \
                                                                                  self.num_clusters, \
                                                                                  self.cluster_train_ratio)
            toc = timeit.default_timer()

            print("Clustering time %d: %4.4f" % (idx, toc - tic))

            for i in range(args.num_clusters):
                self.samples_per_cluster[idx, i] = np.sum(self.clustered_inputs[idx, :] == i, dtype=np.int64)
               
                if (hvd.rank() == 0):
                    print("Cluster %d: %d" % (i, self.samples_per_cluster[idx, i]))
                
#                if (self.samples_per_cluster[idx, i] == 0):
#                    raise ValueError("\n\nCluster %d of file %s has no samples!\n\n" % (i, fpath))

            if (np.sum(self.samples_per_cluster[idx, :]) != self.num_samples):
                raise ValueError("\n\nSamplers per cluster sum: %d, not equal to total num samples: %d\n\n" % (np.sum(self.samples_per_cluster[idx,:]), self.num_samples))

#            print("\n\nDone\n\n")
#            exit(0);

#        for i in range(self.num_clusters):
#            self.cluster_idxs.append(self.clustered_inputs[])


        self.file_idxs = np.random.permutation(np.arange(self.num_files))
        self.current_file = 0
        self.current_sample = 0

        self.barrier = mp.Barrier(self.args.num_data_workers)


        # Set the starting dataset
        self.input_dataset = None
        self.output_dataset = None
        self.cluster_idxs = [None] * self.num_clusters

        self.sampling_idxs = [None] * self.num_clusters
        self.current_sampling_idx = [None] * self.num_clusters


        self.reset_dataset()

    def set_scalers(self, input_scaler, output_scaler):
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        if (not self.is_input_scaled):
            self.input_dataset = self.input_scaler.do_scaling_sample(self.input_dataset)
        else:
            raise ValueError("\n\nBig Clustered Dataset INPUT already scaled. Exiting.\n\n")

        if (not self.is_output_scaled):
            self.output_dataset = self.output_scaler.do_scaling_sample(self.output_dataset)
        else:
            raise ValueError("\n\nBig Clustered Dataset OUTPUT already scaled. Exiting.\n\n")

    # pick a sample within some cluster
    def get_clustered_idx(self, idx):
        
        cluster = int(idx % self.num_clusters)
        file_idx = self.file_idxs[self.current_file]
        num_cluster_samples = self.samples_per_cluster[file_idx, cluster]

        # Rejection sampling
        if (num_cluster_samples == 0):
            bad_iters = 0

            while (num_cluster_samples == 0):
                cluster = np.random.randint(self.num_clusters)
                num_cluster_samples = self.samples_per_cluster[file_idx, cluster]

                if (bad_iters > 100):
                    raise ValueError("\n\n100 successive bad clusters. Exiting.\n\n")

                bad_iters += 1

        # Must randomly select because subsampling and do not want to bias data choice
#        idx_within_cluster = np.random.randint(num_cluster_samples)

        front = self.sampling_idxs[cluster]
#        print("Front: ", type(front))

        back = self.current_sampling_idx[cluster] % num_cluster_samples
        
#        print("Back: ", type(back), back)

        back = int(back)

        idx_within_cluster = front[back]

        # remove chance of revisiting same sample within a sample
#        idx_within_cluster = self.sampling_idxs[cluster][self.current_sampling_idx[cluster] % num_cluster_samples]

        self.current_sampling_idx[cluster] += 1
        
#        if (self.cluster_idxs[cluster].shape[0] != num_cluster_samples):
#            raise ValueError("\n\nRank: %d, Get IDX. New_File_Idx: %d, current_file: %d, Cluster id: %d, CIDS: %d, SPC: %d, SAMPLE: %d\n\n" % (hvd.rank(), file_idx, self.current_file, cluster, self.cluster_idxs[cluster].shape[0], num_cluster_samples, idx_within_cluster))

        # return the original fp sample idx given a cluster and idx_within_cluster
#        return np.arange(self.num_samples)[self.clustered_inputs[file_idx, :] == cluster][idx_within_cluster]
        return self.cluster_idxs[cluster][idx_within_cluster]

    def reset_dataset(self):

        # Create a new permutation
        if (self.current_file == self.num_files):
            self.file_idxs = np.random.permutation(np.arange(self.num_files))
            self.current_file = 0

        del self.input_dataset
        del self.output_dataset

        new_file_idx = self.file_idxs[self.current_file]

        # Load file into memory
        self.input_dataset = np.load(self.input_fpaths[new_file_idx])
        self.output_dataset = np.load(self.output_fpaths[new_file_idx])

        # Reshape data
        self.input_dataset = np.reshape(self.input_dataset, \
                                        self.input_shape)
        self.output_dataset = np.reshape(self.output_dataset, \
                                         self.output_shape)

        # Subset data
        self.input_dataset = self.input_dataset[:, self.input_mask]
        self.output_dataset = self.output_dataset[:, self.output_mask]

        # Scale data
        self.input_dataset = self.input_scaler.do_scaling_sample(self.input_dataset)
        self.output_dataset = self.output_scaler.do_scaling_sample(self.output_dataset)

#        cidxs = mp.Manager().list(range(self.num_clusters))
#        sidxs = mp.Manager().list(range(self.num_clusters))
#        csidx = mp.Manager().list()

#        def set_idx(i):
#            cidxs[i] = np.arange(self.num_samples)[self.clustered_inputs[new_file_idx, :] == i]
#            sidxs[i] = np.random.permutation(np.arange(self.samples_per_cluster[new_file_idx, i], dtype=np.int64))
#            self.current_sampling_idx[i] = 0

#        pool = mp.Pool()

#        for i in range(self.num_clusters):
#            pool.apply_async(set_idx, (i,))

#        pool.close()

#        for i in range(self.num_clusters):
#            self.cluster_idxs[i] = cidxs[i]
#            self.sampling_idxs[i] = sidxs[i]
#            self.current_sampling_idx[i] = 0
        
        # Reset cluster idxs for the new snapshot
        for i in range(self.num_clusters):
            self.cluster_idxs[i] = np.arange(self.num_samples)[self.clustered_inputs[new_file_idx, :] == i]
            self.sampling_idxs[i] = np.random.permutation(np.arange(self.samples_per_cluster[new_file_idx, i], dtype=np.int64))
            self.current_sampling_idx[i] = 0


#            if (self.cluster_idxs[i].shape[0] != self.samples_per_cluster[new_file_idx, i]):
#                raise ValueError("\n\nRank: %d, Reset dataset. New_File_Idx: %d, current_file: %d, Cluster id: %d, CIDS: %d, SPC: %d\n\n" % (hvd.rank(), new_file_idx, self.current_file, i, self.cluster_idxs[i].shape[0], self.samplers_per_cluster[new_file_idx, i]))

    # Fetch a sample
    def __getitem__(self, idx):
     
        if (self.reset and self.current_sample >= (self.num_samples * self.cluster_sample_ratio / hvd.size())):
            pid = self.barrier.wait()
            print("Rank %d, Reset PID: %d" % (hvd.rank(), pid))

            self.current_file += 1
            
            if (pid == 0):
                print("Rank: %d, Entering reset datset on PID %d" % (hvd.rank(), pid))
                self.reset_dataset()

#            self.current_file += 1
            self.current_sample = 0

            print("Rank: %d, PID %d waiting or Done" % (hvd.rank(), pid))
            self.barrier.wait()

        
        self.current_sample += self.args.num_data_workers
#        sample_idx = idx % self.num_samples

        sample_idx = self.get_clustered_idx(idx)

        input_tensor = torch.tensor(self.input_dataset[sample_idx, :], dtype=torch.float32)
        output_tensor = torch.tensor(self.output_dataset[sample_idx, :], dtype=torch.float32)

        return input_tensor, output_tensor


    # Number of samples in dataset
    def __len__(self):
        return int(self.num_files * self.num_samples * self.cluster_sample_ratio)