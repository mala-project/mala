# FP LDOS, Data Loaders

import os, sys
import numpy as np
#import torch.nn as nn
#import torch.nn.functional as F

#import torch.utils.Dataset
import torch.utils.data.distributed
import torch.utils.data
import torch.utils
import horovod.torch as hvd


###-----------------------------------------------------------------------###

# Big Data Dataset
class Big_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data_name):

        fp_data_paths = []
        ldos_data_paths = []

        if (data_name == "train"):
            if (hvd.rank() == 0):
                print("Creating Big Data Train Dataset")

            # Start at snapshot 0
            self.num_snapshots = args.num_snapshots - 2
            snapshot = 0

        elif (data_name == "validation"):
            if (hvd.rank() == 0):
                print("Creating Big Data Validation Dataset")

            # 2nd to last snapshot in the set of num_snapshots
            self.num_snapshots = 1
            snapshot = args.num_snapshots - 2

        elif (data_name == "test"):
            if (hvd.rank() == 0):
                print("Creating Big Data Test Dataset")

            # Last snapshot in the set of num_snapshots
            self.num_snapshots = 1
            snapshot = args.num_snapshots - 1
        else:
            if (hvd.rank() == 0):
                print("\nInvalid Big Datset. Options are ['train', 'validation', or 'test'].\n\n")
            exit(0);

        fp_head = "/%s/%sgcc/%s_fp_%dx%dx%dgrid_%dcomps" % \
                (args.temp, args.gcc, args.material, \
                 args.nxyz, args.nxyz, args.nxyz, args.fp_length)
        ldos_head = "/%s/%sgcc/%s_ldos_%dx%dx%dgrid_%delvls" % \
                (args.temp, args.gcc, args.material, \
                 args.nxyz, args.nxyz, args.nxyz, args.ldos_length)



        for i in range(self.num_snapshots):
            fp_data_paths.append(args.fp_dir + fp_head + \
                    "_snapshot%d.npy" % (snapshot + i))
            ldos_data_paths.append(args.ldos_dir + ldos_head + \
                    "_snapshot%d.npy" % (snapshot + i))
            
        # List of numpy arrays to preserve mmap_mode
        self.fp_dataset = [] 
        self.ldos_dataset = []

        # Load Datasets
        for idx, path in enumerate(fp_data_paths):
            self.fp_dataset.append(np.load(path, mmap_mode="r"))
        for idx, path in enumerate(ldos_data_paths):
            self.ldos_dataset.append(np.load(path, mmap_mode="r"))

        # FP subset and reshape
        fp_idxs = subset_fp(args)
        for i in range(len(self.fp_dataset)):
            self.fp_dataset[i] = self.fp_dataset[i][:,:,:,fp_idxs]

            data_shape = self.fp_dataset[i].shape 

            grid_pts = data_shape[0] * data_shape[1] * data_shape[2]

            self.fp_dataset[i] = np.reshape(self.fp_dataset[i], [grid_pts, data_shape[3]])

            self.fp_dataset[i] = self.fp_dataset[i]

        # !!! Need to modify !!! 
        # Switch args.fp_length -> args.fp_length and args.final_fp_length 
        if (data_name == "test"):
            args.fp_length = data_shape[-1]


        # LDOS subset and reshape
        ldos_idxs = subset_ldos(args)
        for i in range(len(self.ldos_dataset)):
            self.ldos_dataset[i] = self.ldos_dataset[i][:,:,:,ldos_idxs]

            data_shape = self.ldos_dataset[i].shape 

            grid_pts = data_shape[0] * data_shape[1] * data_shape[2]

            self.ldos_dataset[i] = np.reshape(self.ldos_dataset[i], [grid_pts, data_shape[3]])
            
            self.ldos_dataset[i] = self.ldos_dataset[i]

        # !!! Need to modify !!! 
        # Switch args.ldos_length -> args.ldos_length and args.final_ldos_length 
        if (data_name == "test"):
            args.ldos_length = data_shape[-1]
            args.grid_pts = grid_pts

        self.grid_pts = grid_pts

        # Consistency Checks
        if (len(self.fp_dataset) != len(self.ldos_dataset)):
            if (hvd.rank() == 0):
                print("\nError. Num snapshots for fp and ldos inconsistent.\n\n")
            exit(0);

        for i in range(len(self.fp_dataset)):
            if (self.fp_dataset[i].shape[0] != self.ldos_dataset[i].shape[0]):
                if (hvd.rank() == 0):
                    print("\nError. Snapshot %d, FP and LDOS dataset have different number of data points.\n\n" % i)
                exit(0);

            if (self.fp_dataset[0].shape[-1] != self.fp_dataset[i].shape[-1]):
                if (hvd.rank() == 0):
                    print("\nError. Snapshot %d, Fingerprint lengths are not consistent between snapshots.\n\n" % i)
                exit(0);
            
            if (self.ldos_dataset[0].shape[-1] != self.ldos_dataset[i].shape[-1]):
                if (hvd.rank() == 0):
                    print("\nError. Snapshot %d, LDOS lengths are not consistent between snapshots.\n\n" % i)
                exit(0);



    # Fetch a sample
    def __getitem__(self, idx):
      
#        print("idx: ", idx)

        return torch.tensor(self.fp_dataset[idx // self.grid_pts][idx % self.grid_pts, :]).float(), \
               torch.tensor(self.ldos_dataset[idx // self.grid_pts][idx % self.grid_pts, :]).float()

#        print("input shape: ", t1.shape)
#        print("output shape: ", t2.shape)


#        return t1, t2


    # Number of samples in dataset
    def __len__(self):
        return self.num_snapshots * self.grid_pts









###-----------------------------------------------------------------------###

# Compressed Dataset
#class Compressed_Dataset(torch.utils.data.Dataset):
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

#
# RANDOM Dataset
#

def load_data_random(args):
#   args.fp_length = 116
#    args.ldos_length = 128
#    args.dens_length = 1
#    args.lstm_in_length = 10

    if (hvd.rank() == 0):
        print("Begin Load Data for RANDOM")
    
    args.grid_pts = args.nxyz ** 3

    train_pts = int(args.grid_pts * args.train_test_split)
    validation_pts = int((args.grid_pts - train_pts) / 2.0)
    test_pts = args.grid_pts - train_pts - validation_pts


    ldos_random_torch = \
        torch.tensor(np.random.rand(args.grid_pts, args.ldos_length), \
                     dtype=torch.float32)
    
    fp_random_torch = \
        torch.tensor(np.random.rand(args.grid_pts, args.fp_length), \
                                    dtype=torch.float32)

    fp_ldos_dataset = \
        torch.utils.data.TensorDataset(fp_random_torch, ldos_random_torch)

    train_dataset, validation_dataset, test_dataset = \
        torch.utils.data.random_split(fp_ldos_dataset, \
                                     [train_pts, validation_pts, test_pts])

    return (train_dataset, validation_dataset, test_dataset)









###-----------------------------------------------------------------------###

#
# FP_LDOS Dataset
#

def load_data_fp_ldos(args):

    if (hvd.rank() == 0):
        print("Begin Load Data for FP_LDOS")

    # Currently use 
    # 1 snapshot for validation, 
    # 1 snapshot for test, 
    # and the rest for training.
    args.test_snapshot = args.num_snapshots - 1;
    args.validation_snapshot = args.num_snapshots - 2;
    args.num_train_snapshots = args.num_snapshots - 2;

    if (args.num_train_snapshots < 1):
        args.num_train_snapshots = 1
    if (args.validation_snapshot < 0):
        args.validation_snapshot = 0

    # If using water dataset
    if (args.water):
        args.fp_data_fpath = "/%s/%sgcc/~~~~~~~~" % (args.temp, args.gcc)
        args.ldos_data_fpath = "/%s/%sgcc/~~~~~~~~" % (args.temp, args.gcc)

        print("For Josh, water case")
        exit(0);
    # If using Material (Al) dataset
    else:
        args.fp_data_fpath = "/%s/%sgcc/%s_fp_%dx%dx%dgrid_%dcomps" % \
                (args.temp, args.gcc, args.material, args.nxyz, args.nxyz, args.nxyz, args.fp_length)
        args.ldos_data_fpath = "/%s/%sgcc/%s_ldos_%dx%dx%dgrid_%delvls" % \
                (args.temp, args.gcc, args.material, args.nxyz, args.nxyz, args.nxyz, args.ldos_length)

    # Get dimensions of fp/ldos numpy arrays  
    empty_fp_np = np.load(args.fp_dir + args.fp_data_fpath + \
        "_snapshot%d.npy" % (0), mmap_mode='r')
    empty_ldos_np = np.load(args.ldos_dir + args.ldos_data_fpath + \
        "_snapshot%d.npy" % (0), mmap_mode='r')

    fp_shape = empty_fp_np.shape
    ldos_shape = empty_ldos_np.shape

    # Create empty np arrays to store all train snapshots 
    # (FP(input) and LDOS(output)) 
    full_train_fp_np = \
        np.empty(np.insert(fp_shape, 0, args.num_train_snapshots))
    full_train_ldos_np = \
        np.empty(np.insert(ldos_shape, 0, args.num_train_snapshots))

    if (hvd.rank() == 0):
        print("Original Fingerprint shape: ", full_train_fp_np.shape)
        print("Original LDOS shape: ", full_train_ldos_np.shape)
        print("Reading Fingerprint and LDOS dataset")

    hvd.allreduce(torch.tensor(0), name='barrier')
    
    for sshot in range(args.num_train_snapshots):
        print("Rank: %d, Reading train snapshot %d" % (hvd.rank(), sshot))

        full_train_fp_np[sshot, :, :, :, :] = np.load(args.fp_dir + args.fp_data_fpath + \
            "_snapshot%d.npy" % (sshot))

        full_train_ldos_np[sshot, :, :, :, :] = np.load(args.ldos_dir + args.ldos_data_fpath + \
            "_snapshot%d.npy" % (sshot))

        hvd.allreduce(torch.tensor(0), name='barrier')
    
    print("Rank: %d, Reading validation snapshot %d" % (hvd.rank(), args.validation_snapshot))
    validation_fp_np = np.load(args.fp_dir + args.fp_data_fpath + \
        "_snapshot%d.npy" % (args.validation_snapshot))
    validation_ldos_np = np.load(args.ldos_dir + args.ldos_data_fpath + \
        "_snapshot%d.npy" % (args.validation_snapshot))
 
    hvd.allreduce(torch.tensor(0), name='barrier')
    
    print("Rank: %d, Reading test snapshot %d" % (hvd.rank(), args.test_snapshot))
    test_fp_np = np.load(args.fp_dir + args.fp_data_fpath + \
        "_snapshot%d.npy" % (args.test_snapshot))
    test_ldos_np = np.load(args.ldos_dir + args.ldos_data_fpath + \
        "_snapshot%d.npy" % (args.test_snapshot))

    hvd.allreduce(torch.tensor(0), name='barrier')
   

    # Pick subset of FP vector that the user requested
    fp_idxs = subset_fp(args)

    if (hvd.rank() == 0):
        print("Subsetting FP dataset")
        print("FP_idxs: ", fp_idxs)

    full_train_fp_np = full_train_fp_np[:, :, :, :, fp_idxs]
    validation_fp_np = validation_fp_np[:, :, :, fp_idxs]
    test_fp_np = test_fp_np[:, :, :, fp_idxs]


    # Pick subset of LDOS vector that the user requested
    ldos_idxs = subset_ldos(args)

    if (hvd.rank() == 0):
        print("Subsetting LDOS dataset")    
        print("LDOS_idxs: ", ldos_idxs)
        
    full_train_ldos_np = full_train_ldos_np[:, :, :, :, ldos_idxs]
    validation_ldos_np = validation_ldos_np[:, :, :, ldos_idxs]
    test_ldos_np = test_ldos_np[:, :, :, ldos_idxs]


    fp_shape = test_fp_np.shape
    ldos_shape = test_ldos_np.shape

    fp_pts = fp_shape[0] * fp_shape[1] * fp_shape[2]
    ldos_pts = ldos_shape[0] * ldos_shape[1] * ldos_shape[2]

    # Grid inconsistent
    if (fp_pts != ldos_pts):
        print("\n\nError in num grid points: fp_pts %d and ldos_pts %d\n\n" % (fp_pts, ldos_pts));
        exit(0);

    # Bidirection with density prediction
    if (ldos_shape[3] == 1 and (args.model_lstm_network and not args.no_bidirection)):
        print("\n\nError cannot use bidirectional LSTM when predicting densities. Please use unidirectional LSTM or Feedforward only. Exiting.\n\n")
        exit(0);


    args.grid_pts = fp_pts

    args.train_pts = args.grid_pts * args.num_train_snapshots
    args.validation_pts = args.grid_pts
    args.test_pts = args.grid_pts

    # Vector lengths
    args.fp_length = fp_shape[3]
    args.ldos_length = ldos_shape[3]

   
    if (hvd.rank() == 0):
        print("Grid_pts %d" % args.grid_pts)
        print("Train_pts %d" % args.train_pts)
        print("Validation_pts %d" % args.validation_pts)
        print("Test pts %d" % args.test_pts)
        print("Final FP vector length: %d" % args.fp_length)
        print("Final LDOS vector length: %d" % args.ldos_length)
        print("Reshaping Datasets")
    

    # Reshape tensor datasets such that 
    # NUM_SNAPSHOTS x 200 x 200 x 200 x VEC_LEN => (NUM_SNAPSHOTS * 200^3) x VEC_LEN
    full_train_fp_np = full_train_fp_np.reshape([args.train_pts, args.fp_length])
    full_train_ldos_np = full_train_ldos_np.reshape([args.train_pts, args.ldos_length])

    validation_fp_np = validation_fp_np.reshape([args.validation_pts, args.fp_length])
    validation_ldos_np = validation_ldos_np.reshape([args.validation_pts, args.ldos_length])
    
    test_fp_np = test_fp_np.reshape([args.test_pts, args.fp_length])
    test_ldos_np = test_ldos_np.reshape([args.test_pts, args.ldos_length])
    

    # Scale fingerprints
    full_train_fp_np, validation_fp_np, test_fp_np = \
            scale_data(args, "fp", \
                       full_train_fp_np, \
                       validation_fp_np, \
                       test_fp_np, \
                       args.fp_log, \
                       args.fp_row_scaling, \
                       args.fp_norm_scaling,\
                       args.fp_max_only, \
                       args.fp_standard_scaling)

    hvd.allreduce(torch.tensor(0), name='barrier')
    
    # Scale ldos
    full_train_ldos_np, validation_ldos_np, test_ldos_np = \
            scale_data(args, "ldos", \
                       full_train_ldos_np, \
                       validation_ldos_np, \
                       test_ldos_np, \
                       args.ldos_log, \
                       args.ldos_row_scaling, \
                       args.ldos_norm_scaling,\
                       args.ldos_max_only, \
                       args.ldos_standard_scaling)


    # Save modified input/outputs
    if (hvd.rank() == 0 and args.save_training_data):

        print("Saving training data.")

        np.save(args.model_dir + "/full_train_fp_np", full_train_fp_np)
        np.save(args.model_dir + "/validation_fp_np", validation_fp_np)
        np.save(args.model_dir + "/test_fp_np", test_fp_np)

        np.save(args.model_dir + "/full_train_ldos_np", full_train_ldos_np)
        np.save(args.model_dir + "/validation_ldos_np", validation_ldos_np)
        np.save(args.model_dir + "/test_ldos_np", test_ldos_np)



    # Create Pytorch tensors

    hvd.allreduce(torch.tensor(0), name='barrier')

    print("Rank: %d, Creating train tensors" % hvd.rank())
    # Create PyTorch Tensors (and Datasets X/Y) from numpy arrays
    full_train_fp_torch = torch.tensor(full_train_fp_np, dtype=torch.float32)
    hvd.allreduce(torch.tensor(0), name='barrier')
    
    full_train_ldos_torch = torch.tensor(full_train_ldos_np, dtype=torch.float32)
    hvd.allreduce(torch.tensor(0), name='barrier')


    print("Rank: %d, Creating validation tensors" % hvd.rank()) 
    validation_fp_torch = torch.tensor(validation_fp_np, dtype=torch.float32)
    hvd.allreduce(torch.tensor(0), name='barrier')
    
    validation_ldos_torch = torch.tensor(validation_ldos_np, dtype=torch.float32)
    hvd.allreduce(torch.tensor(0), name='barrier')


    print("Rank: %d, Creating test tensors" % hvd.rank())    
    test_fp_torch = torch.tensor(test_fp_np, dtype=torch.float32)
    hvd.allreduce(torch.tensor(0), name='barrier')

    test_ldos_torch = torch.tensor(test_ldos_np, dtype=torch.float32)  
    hvd.allreduce(torch.tensor(0), name='barrier')


    print("Rank: %d, Creating tensor datasets" % hvd.rank())
    # Create fp (inputs) and ldos (outputs) Pytorch Dataset
    train_dataset = torch.utils.data.TensorDataset(full_train_fp_torch, full_train_ldos_torch)
    hvd.allreduce(torch.tensor(0), name='barrier')
    
    validation_dataset = torch.utils.data.TensorDataset(validation_fp_torch, validation_ldos_torch)
    hvd.allreduce(torch.tensor(0), name='barrier')
    
    test_dataset = torch.utils.data.TensorDataset(test_fp_torch, test_ldos_torch)
    hvd.allreduce(torch.tensor(0), name='barrier')


    return (train_dataset, validation_dataset, test_dataset)








###-----------------------------------------------------------------------###

# Normalize FP or LDOS
def scale_data(args, data_name, \
               data_train, data_validation, data_test, \
               apply_log=False, \
               row_scaling=False, \
               norm_scaling=False, max_only=False, \
               standard_scaling=False):

    if (len(data_train.shape) != 2 or len(data_validation.shape) != 2 or len(data_test.shape) != 2):
        if (hvd.rank() == 0):
            print("\nIssue in %s data shape lengths (train, valid, test): (%d, %d, %d), expected length 2. Exiting.\n\n" \
                % (data_name, len(data_train.shape), len(data_validation.shape), len(data_test.shape)))
        exit(0);
   
    # Number of elements in each sample vector
    data_length = data_train.shape[1]

    # Apply log function to the data
    if (apply_log):
        if (hvd.rank() == 0):
            print("Applying Log function to %s" % data_name)   

        train_min = np.min(data_train)
        validation_min = np.min(data_validation)
        test_min = np.min(data_test)
        
        log_shift = np.array([1e-8])

        train_min += log_shift
        validation_min += log_shift
        test_min += log_shift

        if (train_min <= 0.0 or validation_min <= 0.0 or test_min <= 0.0):
            if (hvd.rank() == 0):
                print("\nApplying the log fn to %s fails because there are values <= 0. Exiting.\n\n" % data_name)
            exit(0);

        np.save(args.model_dir + "/log_shift", log_shift)

        data_train      = np.log(data_train + log_shift)
        data_validation = np.log(data_validation + log_shift)
        data_test       = np.log(data_test + log_shift)
        
    # Row vs total scaling
    if (row_scaling and (norm_scaling or standard_scaling)):
        scaling_factors = np.zeros([2, data_length])
        scaling_factors_fname = "/%s_row" % data_name
    else:
        scaling_factors = np.zeros([2, 1])
        scaling_factors_fname = "/%s_total" % data_name

    # Scale features
    if (norm_scaling or standard_scaling):
        # Apply data normalizations
        for row in range(data_length):

            # Row scaling
            if (row_scaling):
                if (standard_scaling):

                    if (args.calc_training_norm_only):
                        data_meanv = np.mean(data_train[:, row])                
                        data_stdv  = np.std(data_train[:, row])
                                                            
                    else: 
                        data_meanv = np.mean(np.concatenate((data_train[:, row], \
                                                             data_validation[:, row], \
                                                             data_test[:, row]), axis=0))
                        data_stdv  = np.std(np.concatenate((data_train[:, row], \
                                                            data_validation[:, row], \
                                                            data_test[:, row]), axis=0))
           
                    data_train[:, row]      = (data_train[:, row] - data_meanv) / data_stdv
                    data_validation[:, row] = (data_validation[:, row] - data_meanv) / data_stdv
                    data_test[:, row]       = (data_test[:, row] - data_meanv) / data_stdv
       
                    scaling_factors[0, row] = data_meanv
                    scaling_factors[1, row] = data_stdv

                else:
                    if (max_only):
                        data_minv = 0
                    else:
                        if (args.calc_training_norm_only):
                            data_minv = np.min(data_train[:, row])
                        else:
                            data_minv = np.min(np.concatenate((data_train[:, row], \
                                                             data_validation[:, row], \
                                                             data_test[:, row]), axis=0))
                    if (args.calc_training_norm_only):
                        data_maxv = np.max(data_train[:, row])
                    else:
                        data_maxv = np.max(np.concatenate((data_train[:, row], \
                                                         data_validation[:, row], \
                                                         data_test[:, row]), axis=0))

                    if (data_maxv - data_minv < 1e-12):
                        print("\nNormalization of %s error. max-min ~ 0. Exiting. \n\n" % data_name)
                        exit(0);
            
                    data_train[:, row]      = (data_train[:, row] - data_minv) / (data_maxv - data_minv)
                    data_validation[:, row] = (data_validation[:, row] - data_minv) / (data_maxv - data_minv)
                    data_test[:, row]       = (data_test[:, row] - data_minv) / (data_maxv - data_minv)
            
            # No row scaling
            else:
                if (standard_scaling):

                    if (args.calc_training_norm_only):
                        data_mean = np.mean(data_train)
                        data_std = np.std(data_train)

                    else:
                        data_mean = np.mean(np.concatenate((data_train, \
                                                          data_validation, \
                                                          data_test), axis=0))
                        data_std  = np.std(np.concatenate((data_train, \
                                                         data_validation, \
                                                         data_test), axis=0))
                     
                    data_train      = (data_train - data_mean) / data_std
                    data_validation = (data_validation - data_mean) / data_std
                    data_test       = (data_test - data_mean) / data_std
                
                    scaling_factors[0, row] = data_mean
                    scaling_factors[1, row] = data_std
                
                else: 
                    if (max_only):
                        data_min = 0
                    else:
                        if (args.calc_training_norm_only):
                            data_min = np.min(data_train)
                        else:
                            data_min = np.min(np.concatenate((data_train, \
                                                            data_validation, \
                                                            data_test), axis=0))  
                    if (args.calc_training_norm_only):
                        data_max = np.max(data_train)
                    else:
                        data_max = np.max(np.concatenate((data_train, \
                                                        data_validation, \
                                                        data_test), axis=0))
                        
                    if (data_max - data_min < 1e-12):
                        print("\nNormalization of %s error. max-min ~ 0. Exiting\n\n" % data_name)
                        exit(0);

                    data_train      = (data_train - data_min) / (data_max - data_min)
                    data_validation = (data_validation - data_min) / (data_max - data_min)
                    data_test       = (data_test - data_min) / (data_max - data_min)

                    scaling_factors[0, row] = data_min
                    scaling_factors[1, row] = data_max


            if (hvd.rank() == 0):
                if (row_scaling):
                    if (standard_scaling):
                        print("%s Row: %g, Mean: %g, Std: %g" % (data_name, row, scaling_factors[0, row], scaling_factors[1, row]))
                    else:
                        print("%s Row: %g, Min: %g, Max: %g" % (data_name, row, scaling_factors[0, row], scaling_factors[1, row]))
                else: 
                    if (standard_scaling):
                        print("%s Total, Mean: %g, Std: %g" % (data_name, scaling_factors[0, 0], scaling_factors[1, 0]))
                    else:
                        print("%s Total, Min: %g, Max: %g" % (data_name, scaling_factors[0, 0], scaling_factors[1, 0]))

            if (row == 0):
                if (row_scaling):
                    if (standard_scaling):
                        scaling_factors_fname += "_standard_mean_std"
                    else:
                        scaling_factors_fname += "_min_max"

                else: 
                    if (standard_scaling):
                        scaling_factors_fname += "_standard_mean_std"
                    else:
                        scaling_factors_fname += "_min_max"

                    # No Row scaling
                    break;
    
    # No LDOS scaling
    else:  
        if (hvd.rank() == 0):
            print("Not applying scaling to %s." % data_name)
        # Identity scaling
        scaling_factors[0,0] = 0.0
        scaling_factors[1,0] = 1.0
        scaling_factors_fname += "_min_max"
 
    # Save normalization coefficients
    np.save(args.model_dir + scaling_factors_fname, scaling_factors)

    hvd.allreduce(torch.tensor(0), name='barrier')

    return (data_train, data_validation, data_test)







###-----------------------------------------------------------------------###

# Take subset of FP inputs
# Options:
# --power-spectrum-only
# --no-coords
# --no-bispectrum
def subset_fp(args): 

    # No subset requested
    if (not (args.no_bispectrum or args.power_spectrum_only or args.no_coords)):
        return np.arange(args.fp_length)


    # Store subset elements
    fp_idxs = np.array([])

    if (args.no_bispectrum):
        if(hvd.rank() == 0):
            print("Removing bispectrum components from the SNAP FP (use coords only).")

        return (np.append(fp_idxs, [0,1,2])).astype(int)


    # User only the power spectrum elements in fingerprints
    if (args.power_spectrum_only):
        if(hvd.rank() == 0):
            print("Using FP power spectrum only.")

        fp_idxs = np.append(fp_idxs, [0,1,2])

        bs_length = args.fp_length - 3
        twojmax = 0

        count = 0
        while (count < bs_length):                
            twojmax += 1
            

            count = 0
            for j1 in range(0, twojmax + 1):
                for j2 in range(0, j1 + 1):
                    for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):

                        if (j >= j1):     
                            if (j2 == 0):
                                fp_idxs = np.append(fp_idxs, count + 3)       
                            count += 1

        print("Rank: %d, twojmax %d, bs_length %d" % (hvd.rank(), twojmax, count))
        
        fp_idxs = fp_idxs.astype(int)
        print("Power Spectrum idx: ", fp_idxs)

        if (count != bs_length):
            print("Error: could not find power spectrum. bs_length = %d and counted_elements = %d" % (bs_length, count))
    else:
        fp_idxs = np.arange(args.fp_length)

    # The first 3 elements in FPs are coords and the rest are bispectrum components
    if (args.no_coords):
        if(hvd.rank() == 0):
            print("Removing X/Y/Z coords from the SNAP FP.")
        fp_idxs = fp_idxs[3:]

    else:
        if(hvd.rank() == 0):
            print("Removing X/Y/Z coords from the SNAP FP.")
        
    return fp_idxs.astype(int)


# Take subset of LDOS outputs
# Current no option available
def subset_ldos(args):

    return np.arange(args.ldos_length)
