from __future__ import print_function

import argparse
import os, sys
import json

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from torchvision import datasets, transforms

import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import horovod.torch as hvd

from datetime import datetime
import timeit
import numpy as np
sys.path.append('./src/')

import data_loaders
import train_networks
import fp_ldos_networks


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
                    help='number of epochs to tolerate no decrease in validation error for early stopping (default: 10)')
parser.add_argument('--optim-patience', type=int, default=5, metavar='N',
                    help='number of epochs to tolerate no decrease in validation error for lr scheduler (default: 5)')
parser.add_argument('--early-stopping', type=float, default=1.0, metavar='ES',
                    help='required validation decrease to not test patience (default: 1.0)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='Optimizer momentum (default: 0.5)')
parser.add_argument('--grad-clip', type=float, default=0.25, metavar='M',
                    help='Optimizer momentum (default: 0.25)')


# Model
parser.add_argument('--model-lstm-network', action='store_true', default=False,
                    help='use the lstm network')
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
                    help='how many neurons in the feedforward layers (default: 300)')
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
parser.add_argument('--fp-length', type=int, default=116, metavar='N',
                    help='number of coefficients in FPs (default: 116)')
parser.add_argument('--ldos-length', type=int, default=128, metavar='N',
                    help='number of energy levels in LDOS  (default: 128)')
parser.add_argument('--no-coords', action='store_true', default=False,
                    help='do not use x/y/z coordinates in fp inputs')
parser.add_argument('--no-bispectrum', action='store_true', default=False,
                    help='do not use bispectrum components in fp inputs (only coordinates)')
parser.add_argument('--calc-training-norm-only', action='store_true', default=False,
                    help='do not use bispectrum components in fp inputs (only coordinates)')
parser.add_argument('--power-spectrum-only', action='store_true', default=False,
                    help='train on only the power spectrum within the fingerprints')
parser.add_argument('--fp-row-scaling', action='store_true', default=False,
                    help='scale the row of fingerprint inputs')
parser.add_argument('--ldos-row-scaling', action='store_true', default=False,
                    help='scale the row of ldos outputs')
parser.add_argument('--fp-norm-scaling', action='store_true', default=False,
                    help='use MinMax normalization so fingerprints are in [0,1] (Default: No scaling)')
parser.add_argument('--ldos-norm-scaling', action='store_true', default=False,
                    help='if using MinMax normalization on ldos outputs, do not scale min (Default: No scaling)')
parser.add_argument('--fp-max-only', action='store_true', default=False,
                    help='if using MinMax normalization on fingerprint inputs, do not scale min (Default: Min set to 0)')
parser.add_argument('--ldos-max-only', action='store_true', default=False,
                    help='if using MinMax normalization on ldos outputs, do not scale min (Default: Min set to 0)')
parser.add_argument('--fp-standard-scaling', action='store_true', default=False,
                    help='standardize the fp inputs to mean 0, std 1 (Default: No scaling)')
parser.add_argument('--ldos-standard-scaling', action='store_true', default=False,
                    help='standardize the ldos outputs to mean 0, std 1 (Default: No scaling)')
parser.add_argument('--fp-log', action='store_true', default=False,
                    help='apply log function to fingerprint inputs before scaling')
parser.add_argument('--ldos-log', action='store_true', default=False,
                    help='apply log function to ldos outputs before scaling')


# Dataset Choice
parser.add_argument('--dataset', type=str, default="random", metavar='DS',
                    help='dataset to train on (ex: "random", "fp_ldos") (default: "random")')
parser.add_argument('--big-data', action='store_true', default=False,
                    help='do not load data into memory (big data case)')
parser.add_argument('--material', type=str, default="Al", metavar='MAT',
                    help='material of snapshots to train on (default: "Al")')
parser.add_argument('--temp', type=str, default="298K", metavar='T',
                    help='temperature of snapshots to train on (default: "298K")')
parser.add_argument('--gcc', type=str, default="2.699", metavar='GCC',
                    help='density of snapshots to train on (default: "2.699")')
parser.add_argument('--num-snapshots', type=int, default=1, metavar='N',
                    help='num snapshots per temp/gcc pair (default: 1)')
parser.add_argument('--water', action='store_true', default=False,
                    help='train on water fp and ldos files')


# Directory Locations
parser.add_argument('--fp-dir', type=str, \
            default="../../training_data/fp_data", \
            metavar="str", help='path to fp data directory (default: ../../training_data/fp_data)')
parser.add_argument('--ldos-dir', type=str, \
            default="../../training_data/ldos_data", \
            metavar="str", help='path to ldos data directory (default: ../../training_data/ldos_data)')
parser.add_argument('--output-dir', type=str, default="./output",
            metavar="str", help='path to output directory (default: ./output_fp_ldos)')

# Tensorboard
parser.add_argument('--tb-ldos-comparisons', type=int, default=4, metavar='N',
                    help='num of ldos comparisons to make for tensorboard visualization (default: 4)')


# Other options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num-data-workers', type=int, default=2, metavar='N',
                    help='number of data workers for async gpu data movement (default: 2)')
parser.add_argument('--num-threads', type=int, default=32, metavar='N',
                    help='number of threads (default: 32)')
parser.add_argument('--num-gpus', type=int, default=1, metavar='N',
                    help='number of gpus (default: 1)')
parser.add_argument('--save-training-data', action='store_true', default=False,
                    help='save the 6 training tensors (input/output: train,validation,test)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
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


if (args.batch_size < hvd.size()):
    print("Changing batch_size from %d to %d (number of ranks)" % (args.batch_size, hvd.size()))
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

    with open(args.model_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


hvd.allreduce(torch.tensor(0), name='barrier')

args.writer = SummaryWriter(args.tb_output_dir)

# num_workers for multiprocessed data loading
kwargs = {'num_workers': args.num_data_workers, 'pin_memory': True} if args.cuda else {}

if (hvd.rank() == 0):
    print("Parser Arguments")
    for arg in vars(args):
        print ("%s: %s" % (arg, getattr(args, arg)))


# Choose dataset

# Random (For Debugging/Performance Scalability)
if (args.dataset == "random"):
    train_dataset, validation_dataset, test_dataset = data_loaders.load_data_random(args)

# FP->LDOS, data fits in memory
elif (args.dataset == "fp_ldos" and not args.big_data):
    train_dataset, validation_dataset, test_dataset = data_loaders.load_data_fp_ldos(args)

# FP->LDOS, data does not fit in memory
elif (args.dataset == "fp_ldos" and args.big_data):
    if(hvd.rank() == 0):
        print("Loading big data case. Disabling all data normalization.")

    train_dataset       = data_loaders.Big_Dataset(args, "train")
    validation_dataset  = data_loaders.Big_Dataset(args, "validation")
    test_dataset        = data_loaders.Big_Dataset(args, "test")
    
else:
    print("\n\nDataset %s is not available. Currently available datasets are (random, fp_ldos)" % args.dataset)
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
validation_sampler = torch.utils.data.distributed.DistributedSampler(
    validation_dataset, num_replicas=hvd.size(), rank=hvd.rank())
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=args.test_batch_size, sampler=validation_sampler, **kwargs)

hvd.allreduce(torch.tensor(0), name='barrier')


print("Rank: %d, Creating test sampler/loader" % hvd.rank())

# TESTING DATA
test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

hvd.allreduce(torch.tensor(0), name='barrier')



# Choose a Model
if (args.model_lstm_network):
    model = fp_ldos_networks.FP_LDOS_LSTM_Net(args)
else:
    model = fp_ldos_networks.FP_LDOS_FF_Net(args)



# Set model hidden state
model.train_hidden = model.init_hidden_train()
model.test_hidden = model.init_hidden_test()

if args.cuda:
    # Move model to GPU.
    model.cuda()
    model.train_hidden = (model.train_hidden[0].cuda(), model.train_hidden[1].cuda())
    model.test_hidden = (model.test_hidden[0].cuda(), model.test_hidden[1].cuda())


# Count number of network parameters
#num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("\nNum Params: %d " % num_params)
#exit(0);

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
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

### TRAIN ####
tot_tic = timeit.default_timer()

train_time = 0; 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.optim_patience, \
        mode="min", factor=0.1, verbose=True)

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
    
    # Global barrier
    #hvd.allreduce(torch.tensor(0), name='barrier')  
    toc = timeit.default_timer()
    
    scheduler.step(epoch_loss)
   
    trainer.set_optimizer(optimizer)

    if (hvd.rank() == 0):
        print("\nEpoch %d of %d, Training time: %3.3f\n" % (epoch, args.epochs, toc - tic))
    train_time += toc - tic
    
    # Early Stopping 
    validate_loss = trainer.validate()

    validate_loss = hvd.allreduce(torch.tensor(validate_loss), name='barrier')
    

    if (validate_loss < prev_validate_loss * args.early_stopping):
        #if (hvd.rank() == 0):
        print("\nValidation loss has decreased from %4.6e to %4.6e\n" % (prev_validate_loss, validate_loss))
        prev_validate_loss = validate_loss
        curr_patience = 0
    else:
        #if (hvd.rank() == 0):
        print("\nValidation loss has NOT decreased enough! (from %4.6e to %4.6e) Patience at %d of %d\n" % \
            (prev_validate_loss, validate_loss, curr_patience + 1, args.early_patience))
        curr_patience += 1
        if (curr_patience >= args.early_patience):
            print("\n\nPatience has been reached! Final validation error %4.6e\n\n" % validate_loss)
            break;

if (hvd.rank() == 0):
    print("\n\nTraining Complete!\n\n")

tic = timeit.default_timer()

test_loss = 0.0
if (hvd.rank() == 0):
    test_loss = trainer.test()

test_loss = hvd.allreduce(torch.tensor(test_loss), name="barrier");

# Global barrier
#hvd.allreduce(torch.tensor(0), name='barrier') 
toc = timeit.default_timer()
    
if (hvd.rank() == 0):

    print("\nSaving model to %s.\n" % (args.model_dir + "/" + args.dataset + "_model"))
    torch.save(model.state_dict(), args.model_dir + "/" + args.dataset + "_model")

    print("Total Epochs %d, Testing time: %3.3f " % (epoch, toc - tic))


tot_toc = timeit.default_timer()

if (hvd.rank() == 0):
    print("\nSuccess!\n")
    print("\n\nTotal train time %4.4f\n\n" % (tot_toc - tot_tic))
