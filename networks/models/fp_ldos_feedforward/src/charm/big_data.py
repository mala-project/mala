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

###-----------------------------------------------------------------------###

# Big Data Dataset for training data that does not fit into memory
class Big_Charm_Dataset(torch.utils.data.Dataset):

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

#        self.input_subset = input_subset
#        self.output_subset = output_subset

        self.num_files = len(input_fpaths)


        self.reset = True
        if (self.num_files == 1):
            self.reset = False

#        print("Num files: %d" % self.num_files)

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
#        print("\n\nDone.\n\n")
#        exit(0);   

        # List of numpy arrays to preserve mmap_mode
#        self.input_datasets = [] 
#        self.output_datasets = []

        # Load Datasets
#        for idx, path in enumerate(input_fpaths):
#            print("Input: %d" % idx)
#            self.input_datasets.append(np.load(path, mmap_mode=mmap_mode))
#            hvd.allreduce(torch.tensor(0), name="barrier")

#        for idx, path in enumerate(output_fpaths):
#            print("Output: %d" % idx)
#            self.output_datasets.append(np.load(path, mmap_mode=mmap_mode))
#            hvd.allreduce(torch.tensor(0), name="barrier")

        # Input subset and reshape
#        for i in range(self.num_files):
#            self.input_datasets[i] = np.reshape(self.input_datasets[i], \
#                                                self.input_shape) 
                                               
            
#            if (input_subset is not None):
#                self.input_datasets[i] = self.input_datasets[i][:, input_subset]

        # Output subset and reshape 
#        for i in range(self.num_files):
#            self.output_datasets[i] = np.reshape(self.output_datasets[i], \
#                                                 self.output_shape)  
                                                
            
#            if (output_subset is not None):
#                self.output_datasets[i] = self.output_datasets[i][:, output_subset]



        self.file_idxs = np.random.permutation(np.arange(self.num_files))
        self.current_file = 0
        self.current_sample = 0

        self.barrier = mp.Barrier(self.args.num_data_workers)
#        self.lock = torch.multiprocessing.Lock()

        # Set the starting dataset

        self.input_dataset = None
        self.output_dataset = None

        self.reset_dataset()
#        self.lock.acquire()
#        self.lock.release()




    def set_scalers(self, input_scaler, output_scaler):
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        if (not self.is_input_scaled):
            self.input_dataset = self.input_scaler.do_scaling_sample(self.input_dataset)
            self.is_input_scaled = True
        else:
            raise ValueError("\n\nBig Clustered Dataset INPUT already scaled. Exiting.\n\n")

        if (not self.is_output_scaled):
            self.output_dataset = self.output_scaler.do_scaling_sample(self.output_dataset)
            self.is_output_scaled = True
        else:
            raise ValueError("\n\nBig Clustered Dataset OUTPUT already scaled. Exiting.\n\n")


    def reset_dataset(self):

        # Clean out memory, because mmap brings those values into memory 
#        del self.input_datasets
#        del self.output_datasets

#        self.input_datasets = []
#        self.output_datasets = []

        # Load Datasets
#        for idx, path in enumerate(input_fpaths):    
#            self.input_datasets[i] = np.load(path, mmap_mode=mmap_mode)

#        for idx, path in enumerate(output_fpaths):      
#            self.output_datasets[i] = np.load(path, mmap_mode=mmap_mode)
     
        # Input/Output reshape
#        for i in range(self.num_files):
#            self.input_datasets[i] = np.reshape(self.input_datasets[i], \
#                                                self.input_shape)  
                                              
#            self.output_datasets[i] = np.reshape(self.output_datasets[i], \
#                                                 self.output_shape)  
          

#        print("Rank: %d, Reset dataset %d of %d for all workers. Current_sample: %d" % \
#                (hvd.rank(), self.current_file + 1, self.num_files, self.current_sample))

#        print("Rank: %d, Parent PID: %d, Current PID: %d" % \
#                (hvd.rank(), os.getppid(), os.getpid()))

        # Lock threads for data reset
#        self.lock.acquire();
        
#        print("Rank: %d, Reset dataset %d of %d for mp-locked workers." % \
#                (hvd.rank(), self.current_file + 1, self.num_files))

        if (self.current_file == self.num_files):
            self.file_idxs = np.random.permutation(np.arange(self.num_files))
            self.current_file = 0

        del self.input_dataset
        del self.output_dataset

        # Load file into memory
        self.input_dataset = np.load(self.input_fpaths[self.file_idxs[self.current_file]])
        self.output_dataset = np.load(self.output_fpaths[self.file_idxs[self.current_file]])

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


#        print("Input fp valuee:", self.input_dataset[3+92+113, :])

#        self.mutex   = mp.Semaphore(1)
#        self.barrier_sema = mp.Semaphore(0)

#        self.barrier = mp.Barrier(self.args.num_data_workers)

#        self.current_file += 1
#        self.current_sample = 0

#        self.lock.release()



    # Fetch a sample
    def __getitem__(self, idx):
     
        # idx to vector location
#        file_idx = idx // self.num_samples
#        sample_idx = idx % self.num_samples

        # read data 
#        sample_input = self.input_datasets[file_idx][sample_idx]
#        sample_output = self.output_datasets[file_idx][sample_idx]

        # subset and scale data
#        scaled_input  = self.input_scaler.do_scaling_sample(sample_input[self.input_subset])
#        scaled_output = self.output_scaler.do_scaling_sample(sample_output[self.output_subset])

        # create torch tensor
#        input_tensor  = torch.tensor(scaled_input, dtype=torch.float32)
#        output_tensor = torch.tensor(scaled_output, dtype=torch.float32)
 
        if (self.reset and self.current_sample >= (self.num_samples / hvd.size())):
            #self.mp_complete = mp.Value('i', False, lock=False)

            #self.lock.acquire()

#            print("Rank %d, Before PID:  %d" % (hvd.rank(), os.getpid()))

            pid = self.barrier.wait()

            print("Rank %d, Reset PID: %d" % (hvd.rank(), pid))

            self.current_file += 1

            #if (not self.mp_complete.value):
            if (pid == 0):

                print("Rank: %d, Entering reset datset on PID %d" % (hvd.rank(), pid))
                self.reset_dataset()

            #self.current_file += 1
            self.current_sample = 0

            print("Rank: %d, PID %d waiting or Done" % (hvd.rank(), pid))

            self.barrier.wait()

#            print("Rank: %d, Current_file Before: %d" % (hvd.rank(), self.current_file))

#            self.mp_complete.value = True

#            self.lock.release()

#            self.barrier.acquire()
#            self.barrier.release()

#            print("Rank: %d, Current_file After: %d" % (hvd.rank(), self.current_file))
        
        self.current_sample += self.args.num_data_workers
        sample_idx = idx % self.num_samples

#        if (self.current_sample % 1000 == 0):
#            print("CurrSample: %d, SampleIDX: %d" % (self.current_sample, sample_idx))

        input_tensor = torch.tensor(self.input_dataset[sample_idx, :], dtype=torch.float32)
        output_tensor = torch.tensor(self.output_dataset[sample_idx, :], dtype=torch.float32)

        return input_tensor, output_tensor


    # Number of samples in dataset
    def __len__(self):
        return self.num_files * self.num_samples


###-----------------------------------------------------------------------###


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



###-----------------------------------------------------------------------###


# Compressed Dataset
#class Big_Compressed_Dataset(torch.utils.data.Dataset):
#
#    def __init__(self, args, data_name, fp_data, ldos_data):
#
#        if (hvd.rank() == 0):
#            print("Creating Big Compressed Dataset:")
#
#        self.args = args
#        self.sample = 0
#       
#        if (args.load_encoder):
#            self.encoder = 0
#        else:
#
#            args.fp_length = fp_data.shape[1]
#
#            self.num_subdim = 2
#            self.ks = 256
#
#            if (args.fp_length % self.num_subdim != 0):
#                print("\n\nPQKMeans division error. %d not a factor of %d. Exiting!\n" % (self.num_subdim, args.fp_length))
#                exit(0)
#
#            self.pqkmeans.encoder.PQEncoder(num_subdim=self.num_subdim, Ks=self.ks
#
#            sample_pts = fp_data.shape[0] * args.compress_fit_ratio
#
#            if (hvd.rank() == 0):
#                print("Begin fitting encoder to subset of data")
#    
#            tic = timeit.default_timer()
#            self.encoder.fit(fp_data[:sample_pts])
#            toc = timeit.default_timer()
#
#            if (hvd.rank() == 0):
#                print("Fit %d samples to %s dataset encoder: %4.4fs" % (sample_pts, data_name, toc - tic))
#            
#            tic
#
#            fp_encode = encoder.transform(fp_data)
#        
#
#            self
#
#
#
#
#        self.cluster_ids = []
#
#        for i in range(args.clusters):
#            self.cluster_ids.append()
#
#       
#
#    def __getitem__(self, idx):
#
#    
#
#        return 0;
#
#    def __len__(self):
#        return 1;



###-----------------------------------------------------------------------###


###-----------------------------------------------------------------------###

class Big_Data_Scaler:

    def __init__(self, 
                 file_paths,
                 num_samples, 
                 data_shape,
                 data_subset=None,
                 element_scaling=False, 
                 standardize=False, 
                 normalize=False, 
                 max_only=False,
                 apply_log=False):

        self.file_paths         = file_paths
        self.num_samples        = num_samples
        self.data_shape         = data_shape
        self.data_subset        = data_subset

        self.element_scaling    = element_scaling
        self.standardize        = standardize
        self.normalize          = normalize
        self.max_only           = max_only
        self.apply_log          = apply_log

        self.no_scaling         = not standardize and not normalize

        print("Calculating scaling factors.")
        self.setup_scaling()

    
    def print_factors(self):
        if (self.no_scaling):
            print("No Scaling")

        if (self.element_scaling):
            if (self.standardize):
                print("Scaling Element Factors (Mean/Std)")
            elif (self.normalize):
                print("Scaling Element Factors (Min/Max)")
        else:
            if (self.standardize):
                print("Scaling Total Factors (Mean/Std)")
            elif (self.normalize):
                print("Scaling Total Factors (Min/Max)")

        for i in range(self.factors.shape[1]):
            print("%d: %4.4f, %4.4f" % (i, self.factors[0, i], self.factors[1, i]))


    # Scale one sample
    def do_scaling_sample(self, x):

        if (self.no_scaling):
            return x

        if (not self.element_scaling):
            if (self.normalize):
                return (x - self.factors[0, 0]) / (self.factors[1, 0] - self.factors[0, 0])
            elif(self.standardize):
                return (x - self.factors[0, 0]) / self.factors[1, 0]
            else:
                raise ValueError("\n\nBad scaling choices.\n\n")
 
        else:
            if (self.normalize):
                return (x - self.factors[0, :]) / (self.factors[1, :] - self.factors[0, :])
            elif (self.standardize):
                return (x - self.factors[0, :]) / self.factors[1, :]
                
            else:
                raise ValueError("\n\nBad scaling choices.\n\n")


    # Undo scaling of one sample
    def undo_scaling_sample(self, x):

        if (self.no_scaling):
            return x

        if (not self.element_scaling):
            if (self.normalize):
                return (x * (self.factors[1, 0] - self.factors[0, 0])) + self.factors[0, 0]
            elif(self.standardize):
                return (x * self.factors[1, 0]) + self.factors[1, 0]
            else:
                raise ValueError("\n\nBad scaling choices.\n\n")
 
        else:
            if (self.normalize):
                return (x * (self.factors[1, :] - self.factors[0, :])) + self.factors[0, :]
            elif (self.standardize):
                return (x * self.factors[1, :]) + self.factors[0, :]
                
            else:
                raise ValueError("\n\nBad scaling choices.\n\n")


    # Scale batch (or full) data
#    def do_scaling_batch(self, x):
#
#        if (self.no_scaling):
#            return x
#
#        if (not self.element_scaling):
#            if (self.normalize):
#                return (x - self.factors[0, 0]) / (self.factors[1, 0] - self.factors[0, 0])
#            elif(self.standardize):
#                return (x - self.factors[0, 0]) / self.factors[1, 0]
#            else:
#                raise ValueError("\n\nBad scaling choices.\n\n")
# 
#        else:
#            if (self.normalize):
#                return (x - self.factors[0, :, None]) / (self.factors[1, :, None] - self.factors[0, :, None])
#            elif (self.standardize):
#                return (x - self.factors[0, :, None]) / self.factors[1, :, None]
#                
#            else:
#                raise ValueError("\n\nBad scaling choices.\n\n")
#
#
#    # Undo scaling of batch (or full) data
#    def undo_scaling_batch(self, x):
#
#        if (self.no_scaling):
#            return x
#
#        if (not self.element_scaling):
#            if (self.normalize):
#                return (x * (self.factors[1, 0] - self.factors[0, 0])) + self.factors[0, 0]
#            elif(self.standardize):
#                return (x * self.factors[1, 0]) + self.factors[1, 0]
#            else:
#                raise ValueError("\n\nBad scaling choices.\n\n")
# 
#        else:
#            if (self.normalize):
#                return (x * (self.factors[1, :, None] - self.factors[0, :, None])) + self.factors[0, :, None]
#            elif (self.standardize):
#                return (x * self.factors[1, :, None]) + self.factors[0, :, None]
#                
#            else:
#                raise ValueError("\n\nBad scaling choices.\n\n")
#



    # Calculate and store scaling factors
    def setup_scaling(self):

        # Factors
        # factors[0,:], Min (normalize) or Mean (standardize)
        # factors[1,:], Max (normalize) or Std  (standardize)

        if (not self.element_scaling):
            self.factors = np.zeros([2, 1])
        else:
            self.factors = np.zeros([2, self.data_subset.size])
       
        if (self.no_scaling):
            print("No scaling. Neither standardize nor normalize scaling choosen. ")
            return;

        sample_count = 0
        count_elems = 0

#        print("Setup")

        for idx, fpath in enumerate(self.file_paths):

            file_data = np.load(fpath)

            # Shape Data
            file_data = np.reshape(file_data, \
                                   np.insert(self.data_shape, \
                                             0, self.num_samples))
            # Subset Data
            if (self.data_subset is not None):
                file_data = file_data[:, self.data_subset]

            # Final data shape
            self.new_shape = np.array(file_data.shape[1:])

            # Total Scaling
            if (not self.element_scaling):
                if (self.normalize):
                    self.calc_normalize(file_data, 0)
                elif (self.standardize):
                    count_elems = file_data.size
                else:
                    raise ValueError("\n\nBad scaling choices.\n\n")

            # Element Scaling
            else:
                for elem in range(np.prod(self.new_shape)):

#                    print("Elem %d" % elem)
#                    elem_idx = np.zeros(self.new_shape, dtype=bool)
#                    elem_slice = np.array([])

                    if (file_data.ndim != 2):
                        raise ValueError("\nScaler only supports [samples x vector] data.\n")

                    if (self.normalize):
                        self.calc_normalize(file_data[:, elem], elem)
                    elif (self.standardize):
                        self.calc_standardize(file_data[:, elem], elem, sample_count)
                    else:
                        raise ValueError("\n\nBad scaling choices.\n\n")

            sample_count += self.num_samples 
        
#        if (self.standardize):
#            self.factors[1, :] = np.sqrt(self.factors[1, :] / standardize_count)

    # Calculate min/max normalization factors for data_batch
    def calc_normalize(self, data, factor_idx):
        
        # Calc data min
        if (not self.max_only):
            data_min = np.min(data)
            if (data_min < self.factors[0, factor_idx]):
                self.factors[0, factor_idx] = data_min

        # Calc data max
        data_max = np.max(data)
        if (data_max > self.factors[1, factor_idx]):
            self.factors[1, factor_idx] = data_max

    # Calculate mean/std normalization factors for data_batch
    def calc_standardize(self, data, factor_idx, count):
        

        #        print(data.size)

#        num_vals = data.size

#        data_mean = np.mean(data)
#        data_std = np.std(data)

#        count += num_vals

#        deltas = np.subtract(data, self.factors[0, factor_idx] * num_vals)
#        self.factors[0, factor_idx] += np.sum(deltas / count)

#        deltas2 = np.subtract(data, self.factors[0, factor_idx] * num_vals)
#        self.factors[1, factor_idx] += np.sum(deltas * deltas2)

        new_mean = np.mean(data)
        new_std = np.std(data)

        num_new_vals = data.size

        old_mean = self.factors[0, factor_idx]
        old_std = self.factors[1, factor_idx]

        self.factors[0, factor_idx] = \
            count / (count + num_new_vals) * old_mean + \
            num_new_vals / (count + num_new_vals) * new_mean

        self.factors[1, factor_idx] = \
            count / (count + num_new_vals) * old_std ** 2 + \
            num_new_vals / (count + num_new_vals) * new_std ** 2 + \
            (count * num_new_vals) / (count + num_new_vals) ** 2 * \
            (old_mean - new_mean) ** 2

        self.factors[1, factor_idx] = np.sqrt(self.factors[1, factor_idx])

#        print(self.factors[0, factor_idx])
#        print(self.factors[1, factor_idx])


###-----------------------------------------------------------------------###

# Normalize FP or LDOS
#def scale_data(args, data_name, \
#               data_train, data_validation, data_test, \
#               apply_log=False, \
#               row_scaling=False, \
#               norm_scaling=False, max_only=False, \
#               standard_scaling=False):
#
#    if (len(data_train.shape) != 2 or len(data_validation.shape) != 2 or len(data_test.shape) != 2):
#        if (hvd.rank() == 0):
#            print("\nIssue in %s data shape lengths (train, valid, test): (%d, %d, %d), expected length 2. Exiting.\n\n" \
#                % (data_name, len(data_train.shape), len(data_validation.shape), len(data_test.shape)))
#        exit(0);
#   
#    # Number of elements in each sample vector
#    data_length = data_train.shape[1]
#
#    # Apply log function to the data
#    if (apply_log):
#        if (hvd.rank() == 0):
#            print("Applying Log function to %s" % data_name)   
#
#        train_min = np.min(data_train)
#        validation_min = np.min(data_validation)
#        test_min = np.min(data_test)
#        
#        log_shift = np.array([1e-8])
#
#        train_min += log_shift
#        validation_min += log_shift
#        test_min += log_shift
#
#        if (train_min <= 0.0 or validation_min <= 0.0 or test_min <= 0.0):
#            if (hvd.rank() == 0):
#                print("\nApplying the log fn to %s fails because there are values <= 0. Exiting.\n\n" % data_name)
#            exit(0);
#
#        np.save(args.model_dir + "/%s_log_shift" % data_name, log_shift)
#
#        data_train      = np.log(data_train + log_shift)
#        data_validation = np.log(data_validation + log_shift)
#        data_test       = np.log(data_test + log_shift)
#        
#    # Row vs total scaling
#    if (row_scaling and (norm_scaling or standard_scaling)):
#        scaling_factors = np.zeros([2, data_length])
#        scaling_factors_fname = "/%s_factor_row" % data_name
#    else:
#        scaling_factors = np.zeros([2, 1])
#        scaling_factors_fname = "/%s_factor_total" % data_name
#
#    # Scale features
#    if (norm_scaling or standard_scaling):
#        # Apply data normalizations
#        for row in range(data_length):
#
#            # Row scaling
#            if (row_scaling):
#                if (standard_scaling):
#
#                    if (args.calc_training_norm_only):
#                        data_meanv = np.mean(data_train[:, row])                
#                        data_stdv  = np.std(data_train[:, row])
#                                                            
#                    else: 
#                        data_meanv = np.mean(np.concatenate((data_train[:, row], \
#                                                             data_validation[:, row], \
#                                                             data_test[:, row]), axis=0))
#                        data_stdv  = np.std(np.concatenate((data_train[:, row], \
#                                                            data_validation[:, row], \
#                                                            data_test[:, row]), axis=0))
#           
#                    data_train[:, row]      = (data_train[:, row] - data_meanv) / data_stdv
#                    data_validation[:, row] = (data_validation[:, row] - data_meanv) / data_stdv
#                    data_test[:, row]       = (data_test[:, row] - data_meanv) / data_stdv
#       
#                    scaling_factors[0, row] = data_meanv
#                    scaling_factors[1, row] = data_stdv
#
#                else:
#                    if (max_only):
#                        data_minv = 0
#                    else:
#                        if (args.calc_training_norm_only):
#                            data_minv = np.min(data_train[:, row])
#                        else:
#                            data_minv = np.min(np.concatenate((data_train[:, row], \
#                                                             data_validation[:, row], \
#                                                             data_test[:, row]), axis=0))
#                    if (args.calc_training_norm_only):
#                        data_maxv = np.max(data_train[:, row])
#                    else:
#                        data_maxv = np.max(np.concatenate((data_train[:, row], \
#                                                         data_validation[:, row], \
#                                                         data_test[:, row]), axis=0))
#
#                    if (data_maxv - data_minv < 1e-12):
#                        print("\nNormalization of %s error. max-min ~ 0. Exiting. \n\n" % data_name)
#                        exit(0);
#            
#                    data_train[:, row]      = (data_train[:, row] - data_minv) / (data_maxv - data_minv)
#                    data_validation[:, row] = (data_validation[:, row] - data_minv) / (data_maxv - data_minv)
#                    data_test[:, row]       = (data_test[:, row] - data_minv) / (data_maxv - data_minv)
#            
#            # No row scaling
#            else:
#                if (standard_scaling):
#
#                    if (args.calc_training_norm_only):
#                        data_mean = np.mean(data_train)
#                        data_std = np.std(data_train)
#
#                    else:
#                        data_mean = np.mean(np.concatenate((data_train, \
#                                                          data_validation, \
#                                                          data_test), axis=0))
#                        data_std  = np.std(np.concatenate((data_train, \
#                                                         data_validation, \
#                                                         data_test), axis=0))
#                     
#                    data_train      = (data_train - data_mean) / data_std
#                    data_validation = (data_validation - data_mean) / data_std
#                    data_test       = (data_test - data_mean) / data_std
#                
#                    scaling_factors[0, row] = data_mean
#                    scaling_factors[1, row] = data_std
#                
#                else: 
#                    if (max_only):
#                        data_min = 0
#                    else:
#                        if (args.calc_training_norm_only):
#                            data_min = np.min(data_train)
#                        else:
#                            data_min = np.min(np.concatenate((data_train, \
#                                                            data_validation, \
#                                                            data_test), axis=0))  
#                    if (args.calc_training_norm_only):
#                        data_max = np.max(data_train)
#                    else:
#                        data_max = np.max(np.concatenate((data_train, \
#                                                        data_validation, \
#                                                        data_test), axis=0))
#                        
#                    if (data_max - data_min < 1e-12):
#                        print("\nNormalization of %s error. max-min ~ 0. Exiting\n\n" % data_name)
#                        exit(0);
#
#                    data_train      = (data_train - data_min) / (data_max - data_min)
#                    data_validation = (data_validation - data_min) / (data_max - data_min)
#                    data_test       = (data_test - data_min) / (data_max - data_min)
#
#                    scaling_factors[0, row] = data_min
#                    scaling_factors[1, row] = data_max
#
#
#            if (hvd.rank() == 0):
#                if (row_scaling):
#                    if (standard_scaling):
#                        print("%s Row: %g, Mean: %g, Std: %g" % (data_name, row, scaling_factors[0, row], scaling_factors[1, row]))
#                    else:
#                        print("%s Row: %g, Min: %g, Max: %g" % (data_name, row, scaling_factors[0, row], scaling_factors[1, row]))
#                else: 
#                    if (standard_scaling):
#                        print("%s Total, Mean: %g, Std: %g" % (data_name, scaling_factors[0, 0], scaling_factors[1, 0]))
#                    else:
#                        print("%s Total, Min: %g, Max: %g" % (data_name, scaling_factors[0, 0], scaling_factors[1, 0]))
#
#            if (row == 0):
#                if (row_scaling):
#                    if (standard_scaling):
#                        scaling_factors_fname += "_standard_mean_std"
#                    else:
#                        scaling_factors_fname += "_min_max"
#
#                else: 
#                    if (standard_scaling):
#                        scaling_factors_fname += "_standard_mean_std"
#                    else:
#                        scaling_factors_fname += "_min_max"
#
#                    # No Row scaling
#                    break;
#    
#    # No LDOS scaling
#    else:  
#        if (hvd.rank() == 0):
#            print("Not applying scaling to %s." % data_name)
#        # Identity scaling
#        scaling_factors[0,0] = 0.0
#        scaling_factors[1,0] = 1.0
#        scaling_factors_fname += "_min_max"
# 
#    # Save normalization coefficients
#    np.save(args.model_dir + scaling_factors_fname, scaling_factors)
#
##    hvd.allreduce(torch.tensor(0), name='barrier')
#
#    return (data_train, data_validation, data_test)
#

###-----------------------------------------------------------------------###





