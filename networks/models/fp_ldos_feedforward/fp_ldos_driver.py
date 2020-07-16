from __future__ import print_function

import faulthandler; faulthandler.enable()


import argparse
import os, sys
import json
import pickle
from glob import glob

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from torchvision import datasets, transforms

import torch.utils.data.distributed
#from torch.utils.tensorboard import SummaryWriter

import horovod.torch as hvd

from datetime import datetime
import timeit
import numpy as np
sys.path.append('./src/')

import data_loaders
import train_networks
import fp_ldos_networks

sys.path.append('./src/charm')

import big_data


# Training settings
parser = argparse.ArgumentParser(description='FP-LDOS Feedforward Network')

# Training
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--train-test-split', type=float, default=.80, metavar='R',
                    help='pecentage of training data to use (default: .80)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--adam', action='store_true', default=False,
                    help='use ADAM optimizer (default: SGD)')
parser.add_argument('--early-patience', type=int, default=10, metavar='N',
                    help='number of epochs to tolerate no decrease in ' \
                         'validation error for early stopping (default: 10)')
parser.add_argument('--optim-patience', type=int, default=5, metavar='N',
                    help='number of epochs to tolerate no decrease in ' \
                         'validation error for lr scheduler (default: 5)')
parser.add_argument('--early-stopping', type=float, default=1.0, metavar='ES',
                    help='required validation decrease to not test patience' + \
                         '(default: 1.0)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='Optimizer momentum (default: 0.5)')
parser.add_argument('--grad-clip', type=float, default=0.25, metavar='M',
                    help='Optimizer momentum (default: 0.25)')


# Model
parser.add_argument('--model-lstm-network', action='store_true', default=False,
                    help='use the lstm network')
parser.add_argument('--model-gnn-network', action='store_true', default=False,
                    help='use the gnn network')
parser.add_argument('--stacked-auto', action='store_true', default=False,
                    help='use the stacked autoencoder layers')
parser.add_argument('--deep-auto', action='store_true', default=False,
                    help='use the deep autoencoder layers')
parser.add_argument('--skip-connection', action='store_true', default=False,
                    help='add skip connection over feedforward layers')
parser.add_argument('--gru', action='store_true', default=False,
                    help='use GRU instead of LSTM')
parser.add_argument('--ff-mid-layers', type=int, default=2, metavar='N',
                    help='how many feedforward layers to use (default: 2)')
parser.add_argument('--ff-width', type=int, default=300, metavar='N',
                    help='how many neurons in the feedforward layers ' + \
                         '(default: 300)')
parser.add_argument('--ae-factor', type=float, default=.8, metavar='N',
                    help='how many feedforward layers to use (default: .8)')
parser.add_argument('--lstm-in-length', type=int, default=10, metavar='N',
                    help='number of stacked lstm layers (default: 10)')
parser.add_argument('--no-bidirection', action='store_true', default=False,
                    help='do not use bidirectional LSTM/RNN')
parser.add_argument('--no-hidden-state', action='store_true', default=False,
                    help='do not use hidden/cell states for the LSTM')


# Inputs/Outputs
parser.add_argument('--nxyz', type=int, default=200, metavar='N',
                    help='num elements along x,y,z dims (default: 200)')
parser.add_argument('--fp-length', type=int, default=94, metavar='N',
                    help='number of coefficients in FPs (default: 94)')
parser.add_argument('--ldos-length', type=int, default=250, metavar='N',
                    help='number of energy levels in LDOS  (default: 250)')
parser.add_argument('--no-coords', action='store_true', default=False,
                    help='do not use x/y/z coordinates in fp inputs')
parser.add_argument('--no-bispectrum', action='store_true', default=False,
                    help='do not use bispectrum components in fp inputs ' + \
                         '(only coordinates)')
parser.add_argument('--calc-training-norm-only', action='store_true', default=False,
                    help='do not use bispectrum components in fp inputs ' + \
                         '(only coordinates)')
parser.add_argument('--power-spectrum-only', action='store_true', default=False,
                    help='train on only the power spectrum within the ' + \
                         'fingerprints')
parser.add_argument('--fp-row-scaling', action='store_true', default=False,
                    help='scale the row of fingerprint inputs')
parser.add_argument('--ldos-row-scaling', action='store_true', default=False,
                    help='scale the row of ldos outputs')
parser.add_argument('--fp-norm-scaling', action='store_true', default=False,
                    help='use MinMax normalization so fingerprints are in ' + \
                         '[0,1] (Default: No scaling)')
parser.add_argument('--ldos-norm-scaling', action='store_true', default=False,
                    help='if using MinMax normalization on ldos outputs, do ' + \
                         'not scale min (Default: No scaling)')
parser.add_argument('--fp-max-only', action='store_true', default=False,
                    help='if using MinMax normalization on fingerprint ' + \
                         'inputs, do not scale min (Default: Min set to 0)')
parser.add_argument('--ldos-max-only', action='store_true', default=False,
                    help='if using MinMax normalization on ldos outputs, do ' + \
                         'not scale min (Default: Min set to 0)')
parser.add_argument('--fp-standard-scaling', action='store_true', default=False,
                    help='standardize the fp inputs to mean 0, std 1 ' + \
                         '(Default: No scaling)')
parser.add_argument('--ldos-standard-scaling', action='store_true', default=False,
                    help='standardize the ldos outputs to mean 0, std 1 ' + \
                         '(Default: No scaling)')
parser.add_argument('--fp-log', action='store_true', default=False,
                    help='apply log function to fingerprint inputs before ' + \
                         'scaling')
parser.add_argument('--ldos-log', action='store_true', default=False,
                    help='apply log function to ldos outputs before scaling')


# Dataset Choice
parser.add_argument('--dataset', type=str, default="random", metavar='DS',
                    help='dataset to train on (ex: "random", "fp_ldos") ' + \
                         '(default: "random")')
parser.add_argument('--big-data', action='store_true', default=False,
                    help='do not load data into memory (big data case)')
parser.add_argument('--big-charm-data', action='store_true', default=False,
                    help='do not load data into memory (big charm data case)')
parser.add_argument('--big-clustered-data', action='store_true', default=False,
                    help='do not load data into memory and cluster data (big charm data case)')
parser.add_argument('--material', type=str, default="Al", metavar='MAT',
                    help='material of snapshots to train on (default: "Al")')
parser.add_argument('--temp', type=str, default="298K", metavar='T',
                    help='temperature of snapshots to train on (default: "298K")')
parser.add_argument('--gcc', type=str, default="2.699", metavar='GCC',
                    help='density of snapshots to train on (default: "2.699")')
parser.add_argument('--num-snapshots', type=int, default=1, metavar='N',
                    help='num snapshots per temp/gcc pair (default: 1)')
parser.add_argument('--no-testing', action='store_true', default=False,
                    help='only train with training/validation snapshots (i.e. test later)')
parser.add_argument('--water', action='store_true', default=False,
                    help='train on water fp and ldos files')
### Clustering
parser.add_argument('--cluster-train-ratio', type=float, default=.05, metavar='N',
                    help='how much training data to use for the compression fit(default: .05)')
parser.add_argument('--cluster-sample-ratio', type=float, default=.1, metavar='N',
                    help='portion of fp-ldos training to use from each cluster (default: .1)')
parser.add_argument('--num-clusters', type=int, default=100, metavar='N',
                    help='number of clusters for fp data (default: 100)')


# Directory Locations
parser.add_argument('--fp-dir', type=str, \
            default="../../training_data/fp_data", \
            metavar="str", help='path to fp data directory ' + \
                                '(default: ../../training_data/fp_data)')
parser.add_argument('--ldos-dir', type=str, \
            default="../../training_data/ldos_data", \
            metavar="str", help='path to ldos data directory ' + \
                                '(default: ../../training_data/ldos_data)')
parser.add_argument('--output-dir', type=str, default="./output",
            metavar="str", help='path to output directory ' + \
                                '(default: ./output_fp_ldos)')

# Tensorboard
parser.add_argument('--tb-ldos-comparisons', type=int, default=4, metavar='N',
                    help='num of ldos comparisons to make for ' + \
                         'tensorboard visualization (default: 4)')


# Other options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-pinned-memory', action='store_true', default=False,
                    help='do not use pinned memory for data staging')
parser.add_argument('--num-data-workers', type=int, default=4, metavar='N',
                    help='number of data workers for async gpu data ' + \
                         'movement (default: 4)')
parser.add_argument('--num-test-workers', type=int, default=16, metavar='N',
                    help='number of data workers for async gpu data ' + \
                         'movement (default: 16)')
parser.add_argument('--num-threads', type=int, default=32, metavar='N',
                    help='number of threads (default: 32)')
parser.add_argument('--num-gpus', type=int, default=1, metavar='N',
                    help='number of gpus (default: 1)')
parser.add_argument('--save-training-data', action='store_true', default=False,
                    help='save the 6 training tensors ' + \
                         '(input/output: train,validation,test)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging ' + \
                         'training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

# Parse User Options
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

# Horovod: initialize MPI library.
hvd.init()
torch.manual_seed(args.seed)

# WELCOME!
if (hvd.rank() == 0):
    print("\n------------------------------------\n")
    print("--  FEEDFORWARD FP-LDOS ML MODEL  --")
    print("\n------------------------------------\n")

    print("Running with %d ranks" % hvd.size())

# Begin Timers
tot_tic = timeit.default_timer()
setup_tic = timeit.default_timer()

if (args.batch_size < hvd.size()):
    print("Changing batch_size from %d to %d (number of ranks)" % \
            (args.batch_size, hvd.size()))
    args.batch_size = hvd.size()

#args.test_batch_size = args.batch_size

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

# Unique timestamp
args.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if (hvd.rank() == 0):
    print("Current Time: %s" % args.timestamp)

# Horovod: limit # of CPU threads to be used per worker.
if (hvd.rank() == 0 and not args.cuda):
    print("Running with %d threads" % (args.num_threads))

torch.set_num_threads(args.num_threads)

# Create output directories if they do not exist
args.model_dir = args.output_dir + "/fp_ldos_dir_" + args.timestamp

args.tb_output_dir = args.model_dir + "/tb_" + args.dataset + "_" + \
        "fp_ldos_" + str(args.nxyz) + "nxyz_" + \
        args.temp + "temp_" + args.gcc + "gcc"

if (hvd.rank() == 0):

    print("Rank: %d" % hvd.rank())

    if not os.path.exists(args.output_dir):
        print("\nCreating output folder %s\n" % args.output_dir)
        os.makedirs(args.output_dir)

    if not os.path.exists(args.model_dir):
        print("\nCreating Model output folder %s\n" % args.model_dir)
        os.makedirs(args.model_dir)
    
    if not os.path.exists(args.tb_output_dir):
        print("\nCreating Tensorboard output folder %s\n" % args.tb_output_dir)
        os.makedirs(args.tb_output_dir)

    with open(args.model_dir + '/commandline_args1.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

hvd.allreduce(torch.tensor(0), name='barrier')

#args.writer = SummaryWriter(args.tb_output_dir)

# num_workers for multiprocessed data loading
if (args.cuda):
    if (args.no_pinned_memory):
        kwargs = {'num_workers': args.num_data_workers, 'pin_memory': False}
        test_kwargs = {'num_workers': args.num_test_workers, 'pin_memory': False}
    else:
        kwargs = {'num_workers': args.num_data_workers, 'pin_memory': True}
        test_kwargs = {'num_workers': args.num_test_workers, 'pin_memory': True}
else:
    kwargs = {}
    test_kwargs = {}

if (hvd.rank() == 0):
    print("Parser Arguments")
    for arg in vars(args):
        print ("%s: %s" % (arg, getattr(args, arg)))


# Choose dataset

# Random (For Debugging/Performance Scalability)
if (args.dataset == "random"):
    train_dataset, validation_dataset, test_dataset = \
            data_loaders.load_data_random(args)

# FP->LDOS, data fits in memory
elif (args.dataset == "fp_ldos" and not args.big_data and not args.big_charm_data):
    train_dataset, validation_dataset, test_dataset = \
            data_loaders.load_data_fp_ldos(args)

# FP->LDOS, data does not fit in memory
elif (args.dataset == "fp_ldos" and args.big_data):
    if(hvd.rank() == 0):
        print("Loading big data case. Disabling all data normalization.")

    train_dataset       = data_loaders.Big_Dataset(args, "train")
    validation_dataset  = data_loaders.Big_Dataset(args, "validation")
    test_dataset        = data_loaders.Big_Dataset(args, "test")
   
elif (args.dataset == "fp_ldos" and args.big_charm_data):
    if(hvd.rank() == 0):
        print("\nLoading big charm data case.")


    args.test_snapshot = args.num_snapshots - 1
    args.validation_snapshot = args.num_snapshots - 2
    args.num_train_snapshots = args.num_snapshots - 2

    if (args.num_train_snapshots < 1):
        args.num_train_snapshots = 1
    if (args.validation_snapshot < 0):
        args.validation_snapshot = 0

    
    args.fp_data_fpath = "/%s/%sgcc/%s_fp_%dx%dx%dgrid_%dcomps" % \
            (args.temp, args.gcc, args.material, args.nxyz, args.nxyz, args.nxyz, args.fp_length)
    args.ldos_data_fpath = "/%s/%sgcc/%s_ldos_%dx%dx%dgrid_%delvls" % \
            (args.temp, args.gcc, args.material, args.nxyz, args.nxyz, args.nxyz, args.ldos_length)


    # gather all fpaths of training data
    train_fp_fpaths = \
        sorted(glob(args.fp_dir + args.fp_data_fpath + \
             "_snapshot[0-%d].npy" % (args.num_train_snapshots - 1)))
    train_ldos_fpaths = \
        sorted(glob(args.ldos_dir + args.ldos_data_fpath + \
             "_snapshot[0-%d].npy" % (args.num_train_snapshots - 1)))

    validation_fp_fpaths = \
        sorted(glob(args.fp_dir + args.fp_data_fpath + \
             "_snapshot[%d].npy" % args.validation_snapshot))
    validation_ldos_fpaths = \
        sorted(glob(args.ldos_dir + args.ldos_data_fpath + \
             "_snapshot[%d].npy" % args.validation_snapshot))

    test_fp_fpaths = \
        sorted(glob(args.fp_dir + args.fp_data_fpath + \
             "_snapshot[%d].npy" % args.test_snapshot))
    test_ldos_fpaths = \
        sorted(glob(args.ldos_dir + args.ldos_data_fpath + \
             "_snapshot[%d].npy" % args.test_snapshot))

    #        print(args.fp_dir + args.fp_data_fpath)

    if (hvd.rank() == 0):
        print("\nTraining data file paths:")
        for i in range(len(train_fp_fpaths)):
            print(train_fp_fpaths[i])
            print(train_ldos_fpaths[i])
    
        print("\nValidation data file paths:")
        for i in range(len(validation_fp_fpaths)):
            print(validation_fp_fpaths[i])
            print(validation_ldos_fpaths[i])
    
        print("\nTest data file paths:")
        for i in range(len(test_fp_fpaths)):
            print(test_fp_fpaths[i])
            print(test_ldos_fpaths[i])


    # input/output sample shape in the file
    input_shape = np.array([args.fp_length])
    output_shape = np.array([args.ldos_length])

    # input/output subset
    input_subset = data_loaders.subset_fp(args)
    output_subset = data_loaders.subset_ldos(args)

    # input/output scaler options
    input_scaler_kwargs  = {'element_scaling': args.fp_row_scaling, \
                            'standardize':     args.fp_standard_scaling, \
                            'normalize':       args.fp_norm_scaling, \
                            'max_only':        args.fp_max_only, \
                            'apply_log':       args.fp_log}

    output_scaler_kwargs = {'element_scaling': args.ldos_row_scaling, \
                            'standardize':     args.ldos_standard_scaling, \
                            'normalize':       args.ldos_norm_scaling, \
                            'max_only':        args.ldos_max_only, \
                            'apply_log':       args.ldos_log}

    hvd.allreduce(torch.tensor(0), name='barrier')

    if (hvd.rank() == 0):
        print("\n\nCreating train Charm dataset")
    hvd.allreduce(torch.tensor(0), name='barrier')

    if (args.big_clustered_data):
        train_dataset       = big_data.Big_Charm_Clustered_Dataset(args, \
                                                                   train_fp_fpaths, \
                                                                   train_ldos_fpaths, \
                                                                   args.nxyz ** 3, \
                                                                   input_shape, \
                                                                   output_shape, \
                                                                   input_subset, \
                                                                   output_subset, \
                                                                   input_scaler_kwargs, \
                                                                   output_scaler_kwargs) #, \
                                                                   #no_reset=False)
    else:
        train_dataset       = big_data.Big_Charm_Dataset(args, \
                                                         train_fp_fpaths, \
                                                         train_ldos_fpaths, \
                                                         args.nxyz ** 3, \
                                                         input_shape, \
                                                         output_shape, \
                                                         input_subset, \
                                                         output_subset, \
                                                         input_scaler_kwargs, \
                                                         output_scaler_kwargs) #, \
                                                         #no_reset=False)

    hvd.allreduce(torch.tensor(0), name='barrier')
    
    if (hvd.rank() == 0):
        print("\n\nCreating validation Charm dataset")
    hvd.allreduce(torch.tensor(0), name='barrier')

    validation_dataset  = big_data.Big_Charm_Dataset(args, \
                                                     validation_fp_fpaths, \
                                                     validation_ldos_fpaths, \
                                                     args.nxyz ** 3, \
                                                     input_shape, \
                                                     output_shape, \
                                                     input_subset, \
                                                     output_subset) #, \
                                                     #mmap_mode=None)
    # Avoid recalculation of data scaler
#    validation_dataset.input_scaler = train_dataset.input_scaler
#    validation_dataset.output_scaler = train_dataset.output_scaler
 
    validation_dataset.set_scalers(train_dataset.input_scaler, train_dataset.output_scaler)

    hvd.allreduce(torch.tensor(0), name='barrier')

    if (hvd.rank() == 0):
        print("\n\nCreating test Charm dataset")
    hvd.allreduce(torch.tensor(0), name='barrier')

    if (not args.no_testing):
        test_dataset        = big_data.Big_Charm_Dataset(args, \
                                                         test_fp_fpaths, \
                                                         test_ldos_fpaths, \
                                                         args.nxyz ** 3, \
                                                         input_shape, \
                                                         output_shape, \
                                                         input_subset, \
                                                         output_subset) #, \
                                                         #mmap_mode=None)
    
#        test_dataset.input_scaler = train_dataset.input_scaler
#        test_dataset.output_scaler = train_dataset.output_scaler
        
        test_dataset.set_scalers(train_dataset.input_scaler, train_data.output_scaler)
    
    else:
        test_dataset = None

    hvd.allreduce(torch.tensor(0), name='barrier')
    
    # Set new fp/ldos lengths
    args.fp_length   = len(input_subset)
    args.ldos_length = len(output_subset)
    args.grid_pts = args.nxyz ** 3

    torch.save(train_dataset.input_scaler, args.model_dir + "/charm_input_scaler.pth")
    torch.save(train_dataset.output_scaler, args.model_dir + "/charm_output_scaler.pth")

else:
    print("\n\nDataset %s is not available. Currently available " + \
          "datasets are (random, fp_ldos)" % args.dataset)
    exit(0)

hvd.allreduce(torch.tensor(0), name='barrier')




print("Rank: %d, Creating train sampler/loader" % hvd.rank())

# TRAINING DATA
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

hvd.allreduce(torch.tensor(0), name='barrier')


print("Rank: %d, Creating validation sampler/loader" % hvd.rank())

# VALIDATION DATA
validation_sampler = \
        torch.utils.data.distributed.DistributedSampler( \
            validation_dataset, \
            num_replicas=hvd.size(), \
            rank=hvd.rank())
validation_loader = \
        torch.utils.data.DataLoader( \
            validation_dataset, \
            batch_size=args.test_batch_size, \
            sampler=validation_sampler, \
            **test_kwargs)

hvd.allreduce(torch.tensor(0), name='barrier')

if (not args.no_testing):
    print("Rank: %d, Creating test sampler/loader" % hvd.rank())

    # TESTING DATA
    test_sampler = \
            torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_loader = \
            torch.utils.data.DataLoader( \
                test_dataset, \
                batch_size=args.test_batch_size, \
                sampler=test_sampler, \
                **test_kwargs)

else:
    if (hvd.rank() == 0):
        print("Skip creating the test sampler/loader.")

    test_sampler = None
    test_loader = None

hvd.allreduce(torch.tensor(0), name='barrier')


# Choose and create a Model
if (args.model_lstm_network):
    model = fp_ldos_networks.FP_LDOS_LSTM_Net(args)
elif (args.model_gnn_network):
    raise ValueError("For James...")
else:
    model = fp_ldos_networks.FP_LDOS_FF_Net(args)

# Set model hidden state
model.train_hidden = model.init_hidden_train()
model.test_hidden = model.init_hidden_test()

if args.cuda:
    # Move model to GPU.
    model.cuda()
    model.train_hidden = \
            (model.train_hidden[0].cuda(), model.train_hidden[1].cuda())
    model.test_hidden = \
            (model.test_hidden[0].cuda(), model.test_hidden[1].cuda())



# Count number of network parameters
if (hvd.rank() == 0):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nNum Model Params: %d " % num_params)

# Horovod: scale learning rate by the number of GPUs.
if (args.adam):
    optimizer = optim.Adam(model.parameters(), lr=args.lr * hvd.size())

else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                          momentum=args.momentum, nesterov=True)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce \
        else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)
  

# Save args once model is created (txt and pkl formats)
if (hvd.rank() == 0):
    # Plain txt, after pre-processing
    with open(args.model_dir + '/commandline_args2.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Python obj pickle for reloading later
    args_file = args.model_dir + "/commandline_args.pth"
#    afile = open(args_file, 'wb')
#    pickle.dump(args, afile)
#    afile.close()

    torch.save(args, args_file)
    
setup_toc = timeit.default_timer()

### TRAIN ####

# Timers 
train_time = 0.0; 
validation_time = 0.0; 
test_time = 0.0; 

scheduler = \
        torch.optim.lr_scheduler.ReduceLROnPlateau( \
            optimizer, \
            patience=args.optim_patience, \
            mode="min", \
            factor=0.1, \
            threshold=1.0-args.early_stopping, \
            verbose=True)

epoch_loss = 0.0
prev_validate_loss = 1e16
validate_loss = 0.0

curr_patience = 0

if (hvd.rank() == 0):
    print("\n\nBegin Training!\n\n")

# Set training objs

trainer = train_networks.Net_Trainer(args)
trainer.set_model(model)
trainer.set_optimizer(optimizer)
trainer.set_data_samplers(train_sampler, validation_sampler, test_sampler)
trainer.set_data_loaders(train_loader, validation_loader, test_loader)


for epoch in range(1, args.epochs + 1):

    tic = timeit.default_timer()
    epoch_loss = trainer.train(epoch)
    toc = timeit.default_timer()

    if (hvd.rank() == 0):
        print("\nEpoch %d of %d, Training time: %3.3f\n" % \
                (epoch, args.epochs, toc - tic))
    train_time += toc - tic
    
    # Early Stopping 
    tic = timeit.default_timer()
    validate_loss = trainer.validate()
    toc = timeit.default_timer()
    
    if (hvd.rank() == 0):
        print("\nEpoch %d of %d, Validation time: %3.3f\n" % \
                (epoch, args.epochs, toc - tic))
    validation_time += toc - tic

    validate_loss = \
            hvd.allreduce(torch.tensor(validate_loss), \
                          name='validation_loss_reduction')
   
    # Early stopping w/ validation loss
    if (validate_loss < prev_validate_loss * args.early_stopping):
        #if (hvd.rank() == 0):
        print("Validation loss has decreased from %4.6e to %4.6e" % \
                (prev_validate_loss, validate_loss))
        prev_validate_loss = validate_loss
        curr_patience = 0
    else:
        #if (hvd.rank() == 0):
        print("Validation loss has NOT decreased enough! " + \
              "(from %4.6e to %4.6e) Patience at %d of %d" % \
                    (prev_validate_loss, \
                     validate_loss, \
                     curr_patience + 1, \
                     args.early_patience))

        curr_patience += 1
        if (curr_patience >= args.early_patience):
            print("\n\nPatience has been reached! " + \
                  "Final validation error %4.6e\n\n" % validate_loss)
            break;

    # LR scheduler decrease if no sufficient decrease w/ validation loss
    scheduler.step(validate_loss) 
    trainer.set_optimizer(optimizer)

if (hvd.rank() == 0):
    print("\n\nTraining Complete!\n\n")

tic = timeit.default_timer()

if (not args.no_testing):
    test_loss = 0.0
    if (hvd.rank() == 0):
        test_loss = trainer.test()

    test_loss = hvd.allreduce(torch.tensor(test_loss), name="test_loss_reduction");

toc = timeit.default_timer()

test_time = toc - tic

if (hvd.rank() == 0):

    model_fpath = args.model_dir + "/" + args.dataset + "_model.pth"

    print("\nSaving model to %s.\n" % (model_fpath))
#    torch.save(model.state_dict(), model_fpath)
    torch.save(model, model_fpath)

#    print("Total Epochs %d, Testing time: %3.3f " % (epoch, toc - tic))


tot_toc = timeit.default_timer()

if (hvd.rank() == 0):
    print("\nSuccess!\n")
    print("\n\n---Results---\n")
    print("Total Epochs: \t\t%d" % (epoch))
    print("Total Ranks: \t\t%d" % (hvd.size()))
    print("--------------")
    print("| Total Time: \t\t%4.4f" % (tot_toc - tot_tic))
    print("| Setup Time: \t\t%4.4f" % (setup_toc - setup_tic))
    print("| Train Time: \t\t%4.4f" % (train_time))
    print("| Valid Time: \t\t%4.4f" % (validation_time))
    print("|  Test Time: \t\t%4.4f\n\n" % (test_time))


hvd.shutdown()

