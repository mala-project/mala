from __future__ import print_function

import argparse
import os, sys

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

sys.path.append('../utils/')

import ldos_calc


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
parser.add_argument('--early-patience', type=int, default=10, metavar='N',
                    help='number of epochs to tolerate no decrease in validation error (default: 10)')
parser.add_argument('--optim-patience', type=int, default=5, metavar='N',
                    help='number of epochs to tolerate no decrease in validation error (default: 5)')
parser.add_argument('--early-stopping', type=float, default=1.0, metavar='ES',
                    help='required validation decrease to not test patience (default: 1.0)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='Optimizer momentum (default: 0.5)')

# Model
parser.add_argument('--dataset', type=str, default="random", metavar='DS',
                    help='dataset to train on (ex: "random", "fp_ldos") (default: "random")')
parser.add_argument('--model', type=int, default=5, metavar='N',
                    help='model choice (default: 5)')
parser.add_argument('--nxyz', type=int, default=40, metavar='N',
                    help='num elements along x,y,z dims (default: 40)')
parser.add_argument('--no-coords', action='store_true', default=False,
                    help='do not use x/y/z coordinates in fp inputs')
parser.add_argument('--no-bispectrum', action='store_true', default=False,
                    help='do not use bispectrum components in fp inputs (only coordinates)')
parser.add_argument('--temp', type=str, default="298K", metavar='T',
                    help='temperature of snapshot to train on (default: "298K")')
parser.add_argument('--gcc', type=str, default="2.699", metavar='GCC',
                    help='density of snapshot to train on (default: "2.699")')
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
parser.add_argument('--num-threads', type=int, default=32, metavar='N',
                    help='number of threads (default: 32)')
parser.add_argument('--num-gpus', type=int, default=1, metavar='N',
                    help='number of gpus (default: 1)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
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
args.timestamp = datetime.now().strftime("%c")
if (hvd.rank() == 0):
    print("Current Time: %s" % args.timestamp)

# Horovod: limit # of CPU threads to be used per worker.
if (hvd.rank() == 0 and not args.cuda):
    print("Running with %d threads" % (args.num_threads))

torch.set_num_threads(args.num_threads)

# Create output directories if they do not exist
args.model_dir = args.output_dir + "/" + args.timestamp

args.tb_output_dir = args.model_dir + "/tb_" + args.dataset + "_" + \
        str(args.model) + "model_" + str(args.nxyz) + "nxyz_" + \
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

hvd.allreduce(torch.tensor(0), name='barrier')

args.writer = SummaryWriter(args.tb_output_dir)

# num_workers for multiprocessed data loading
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

if (hvd.rank() == 0):
    print("Parser Arguments")
    for arg in vars(args):
        print ("%s: %s" % (arg, getattr(args, arg)))

# Choose a Model

# Model 1: Density estimation with activations
# Model 2: Density estimation without activations
# Model 3: LDOS estimation without LSTM
# Model 4: LDOS estimation with LSTM
# Model 5: LDOS estimation with bidirectional LSTM

if (args.model != 0):
    model_choice = args.model

# Other model params
dens_length = 1
lstm_in_length = 10

if(hvd.rank() == 0):
    print("Running with %d rank(s)" % hvd.size())
    if (model_choice == 1):
        print("\nBuilding Density estimation model with activations\n")
    elif (model_choice == 2):
        print("\nBuilding Density estimation model without activations\n")
    elif (model_choice == 3):
        print("\nBuilding LDOS estimation model without LSTM\n")
    elif (model_choice == 4):
        print("\nBuilding LDOS estimation model with LSTM\n")
    elif (model_choice == 5):
        print("\nBuilding LDOS estimation model with bidirectional LSTM\n")
    else:
        print("Error in model choice");
        exit();

hvd.allreduce(torch.tensor(0), name='barrier')

# Set up model outputs
if (args.dataset == "random"):

    args.fp_length = 116
    args.ldos_length = 128
    args.dens_length = 1
    args.lstm_in_length = 10

    args.grid_pts = args.nxyz ** 3

    train_pts = int(args.grid_pts * args.train_test_split)
    validation_pts = int((args.grid_pts - train_pts) / 2.0)
    test_pts = args.grid_pts - train_pts - validation_pts

    # Density models
    if (model_choice == 1 or model_choice == 2):
        ldos_random_torch = torch.tensor(np.random.rand(args.grid_pts, dens_length), dtype=torch.float32)
    # LDOS models
    elif (model_choice == 3 or model_choice == 4 or model_choice == 5):
        ldos_random_torch = torch.tensor(np.random.rand(args.grid_pts, args.ldos_length), dtype=torch.float32)
    else:
        print("Error in model choice");
        exit();

    fp_random_torch = torch.tensor(np.random.rand(args.grid_pts, args.fp_length), dtype=torch.float32)

    fp_ldos_dataset = torch.utils.data.TensorDataset(fp_random_torch, ldos_random_torch)

    train_dataset, validation_dataset, test_dataset = \
torch.utils.data.random_split(fp_ldos_dataset, [train_pts, validation_pts, test_pts])

elif (args.dataset == "fp_ldos"):
    # FP_LDOS datasets should only be used for LDOS prediction (i.e. not Model 1 or 2)
    if (model_choice != 3 and model_choice != 4 and model_choice != 5):
        print("Error in model choice with fp_ldos dataset. Please use model = {3, 4, or 5}");
        exit();

    # Currently use 1 snapshot for validation, 1 snapshot for test, and the rest for training.
    args.test_snapshot = args.num_snapshots - 1;
    args.validation_snapshot = args.num_snapshots - 2;
    args.num_train_snapshots = args.num_snapshots - 2;

    if (args.num_train_snapshots < 1):
        args.num_train_snapshots = 1
    if (args.validation_snapshot < 0):
        args.validation_snapshot = 0

    # Get dimensions of fp/ldos numpy arrays  
    empty_fp_np = np.load(args.fp_dir + \
        "/%s/%sgcc/Al_fingerprint_snapshot%d.npy" % (args.temp, args.gcc, 0), mmap_mode='r')
    empty_ldos_np = np.load(args.ldos_dir + \
        "/%s/%sgcc/ldos_200x200x200grid_128elvls_snapshot%d.npy" % (args.temp, args.gcc, 0), mmap_mode='r')

    fp_shape = empty_fp_np.shape
    ldos_shape = empty_ldos_np.shape

    # Create empty np arrays to store all train snapshots (FP(input) and LDOS(output)) 
    full_train_fp_np = np.empty(np.insert(fp_shape, 0, args.num_train_snapshots))
    full_train_ldos_np = np.empty(np.insert(ldos_shape, 0, args.num_train_snapshots))

    print(full_train_fp_np.shape)
    print(full_train_ldos_np.shape)

    print("Reading Fingerprint and LDOS dataset")
    
    hvd.allreduce(torch.tensor(0), name='barrier')
    
    for sshot in range(args.num_train_snapshots):
        print("Rank: %d, Reading train snapshot %d" % (hvd.rank(), sshot))

 #       temp_fp = \
        full_train_fp_np[sshot, :, :, :] = np.load(args.fp_dir + \
            "/%s/%sgcc/Al_fingerprint_snapshot%d.npy" % (args.temp, args.gcc, sshot))

#        temp_ldos = \
        full_train_ldos_np[sshot, :, :, :] = np.load(args.ldos_dir + \
            "/%s/%sgcc/ldos_200x200x200grid_128elvls_snapshot%d.npy" % (args.temp, args.gcc, sshot))

        hvd.allreduce(torch.tensor(0), name='barrier')
    
    print("Rank: %d, Reading validation snapshot %d" % (hvd.rank(), args.validation_snapshot))
    validation_fp_np = np.load(args.fp_dir + \
        "/%s/%sgcc/Al_fingerprint_snapshot%d.npy" % (args.temp, args.gcc, args.validation_snapshot))
    validation_ldos_np = np.load(args.ldos_dir + \
        "/%s/%sgcc/ldos_200x200x200grid_128elvls_snapshot%d.npy" % (args.temp, args.gcc, args.validation_snapshot))
 
    hvd.allreduce(torch.tensor(0), name='barrier')
    
    print("Rank: %d, Reading test snapshot %d" % (hvd.rank(), args.test_snapshot))
    test_fp_np = np.load(args.fp_dir + \
        "/%s/%sgcc/Al_fingerprint_snapshot%d.npy" % (args.temp, args.gcc, args.test_snapshot))
    test_ldos_np = np.load(args.ldos_dir + \
        "/%s/%sgcc/ldos_200x200x200grid_128elvls_snapshot%d.npy" % (args.temp, args.gcc, args.test_snapshot))

    hvd.allreduce(torch.tensor(0), name='barrier')
    
    # Pick subset of FP vector is user requested
    # The first 3 elements in FPs are coords and the rest are bispectrum components
    if (args.no_coords):
        # Remove first 3 elements of fp's (x/y/z coords)
        full_train_fp_np = full_train_fp_np[:, :, :, :, 3:]
        validation_fp_np = validation_fp_np[:, :, :, 3:]
        test_fp_np = test_fp_np[:, :, :, 3:]

    elif (args.no_bispectrum):
        fp_np = fp_np[:, :, :, :3]
        validation_fp_np = validation_fp_np[:, :, :, :3]
        test_fp_np = test_fp_np[:, :, :, :3]

    fp_shape = test_fp_np.shape
    ldos_shape = test_ldos_np.shape

    fp_pts = fp_shape[0] * fp_shape[1] * fp_shape[2]
    ldos_pts = ldos_shape[0] * ldos_shape[1] * ldos_shape[2]

    if (fp_pts != ldos_pts):
        print("\n\nError in num grid points: fp_pts %d and ldos_pts %d\n\n" % (fp_pts, ldos_pts));
        exit(0)

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
        print("Fingerprint vector length: %d" % args.fp_length)
        print("LDOS vector length: %d" % args.ldos_length)

    # Reshape tensor datasets such that 
    # NUM_SNAPSHOTS x 200 x 200 x 200 x VEC_LEN => (NUM_SNAPSHOTS * 200^3) x VEC_LEN
    full_train_fp_np = full_train_fp_np.reshape([args.train_pts, args.fp_length])
    full_train_ldos_np = full_train_ldos_np.reshape([args.train_pts, args.ldos_length])

    validation_fp_np = validation_fp_np.reshape([args.validation_pts, args.fp_length])
    validation_ldos_np = validation_ldos_np.reshape([args.validation_pts, args.ldos_length])
    
    test_fp_np = test_fp_np.reshape([args.test_pts, args.fp_length])
    test_ldos_np = test_ldos_np.reshape([args.test_pts, args.ldos_length])
    

    # Row scaling of features. Currently, scale by mean and std (Disabled: to Uniform [0,1])
    for row in range(args.fp_length):

        #meanv = np.mean(fp_np[row, :])
#        fp_maxv  = np.max(fp_np[row, :])
#        fp_minv  = np.min(fp_np[row, :])
        
        fp_meanv = np.mean([full_train_fp_np[row,:], validation_fp_np[row, :], test_fp_np[row, :]])
        fp_stdv  = np.std([full_train_fp_np[row,:], validation_fp_np[row, :], test_fp_np[row, :]])

#        fp_np[row, :] = (fp_np[row, :] - fp_minv) / (fp_maxv - fp_minv)
#        fp_np[row, :] = fp_np[row, :] / fp_maxv
        full_train_fp_np[row, :] = (full_train_fp_np[row, :] - fp_meanv) / fp_stdv
        validation_fp_np[row, :] = (validation_fp_np[row, :] - fp_meanv) / fp_stdv
        test_fp_np[row, :] = (test_fp_np[row, :] - fp_meanv) / fp_stdv

        if (hvd.rank() == 0):
#            print("FP Row: %g, Min: %g, Avg: %g, Max: %g" % (row, np.min(fp_np[row, :]), np.mean(fp_np[row, :]), np.max(fp_np[row, :])))
            print("FP Row: %g, Mean: %g, Std: %g" % (row, fp_meanv, fp_stdv))

    # Row scaling of outputs
    for row in range(args.ldos_length):

        #meanv = np.mean(fp_np[row, :])
#        ldos_maxv  = np.max(ldos_np[row, :])
#        ldos_minv  = np.min(ldos_np[row, :])

        ldos_meanv = np.mean([full_train_ldos_np[row,:], validation_ldos_np[row, :], test_ldos_np[row, :]])
        ldos_stdv  = np.std([full_train_ldos_np[row,:], validation_ldos_np[row, :], test_ldos_np[row, :]])
        
#        ldos_np[row, :] = (ldos_np[row, :] - ldos_minv) / (ldos_maxv - ldos_minv)
#        fp_np[row, :] = fp_np[row, :] / ldos_maxv
        full_train_ldos_np[row, :] = (full_train_ldos_np[row, :] - ldos_meanv) / ldos_stdv
        validation_ldos_np[row, :] = (validation_ldos_np[row, :] - ldos_meanv) / ldos_stdv
        test_ldos_np[row, :] = (test_ldos_np[row, :] - ldos_meanv) / ldos_stdv

        if (hvd.rank() == 0):
#            print("LDOS Row: %g, Min: %g, Avg: %g, Max: %g" % (row, np.min(ldos_np[row, :]), np.mean(ldos_np[row, :]), np.max(ldos_np[row, :])))
            print("LDOS Row: %g, Mean: %g, Std: %g" % (row, ldos_meanv, ldos_stdv))
    

    hvd.allreduce(torch.tensor(0), name='barrier')
    print("Rank: %d, Creating train tensor" % hvd.rank())

    # Create PyTorch Tensors (and Datasets X/Y) from numpy arrays
    full_train_fp_torch = torch.tensor(full_train_fp_np, dtype=torch.float32)
    full_train_ldos_torch = torch.tensor(full_train_ldos_np, dtype=torch.float32)

    hvd.allreduce(torch.tensor(0), name='barrier')
    print("Rank: %d, Creating validation tensor" % hvd.rank())
    
    validation_fp_torch = torch.tensor(validation_fp_np, dtype=torch.float32)
    validation_ldos_torch = torch.tensor(validation_ldos_np, dtype=torch.float32)

    hvd.allreduce(torch.tensor(0), name='barrier')
    print("Rank: %d, Creating test tensor" % hvd.rank())
    
    test_fp_torch = torch.tensor(test_fp_np, dtype=torch.float32)
    test_ldos_torch = torch.tensor(test_ldos_np, dtype=torch.float32)
    

    # Create fp (inputs) and ldos (outputs) Pytorch Dataset and apply random split
    train_dataset = torch.utils.data.TensorDataset(full_train_fp_torch, full_train_ldos_torch)
    validation_dataset = torch.utils.data.TensorDataset(validation_fp_torch, validation_ldos_torch)
    test_dataset = torch.utils.data.TensorDataset(test_fp_torch, test_ldos_torch)


else:
    print("\n\nDataset %s is not available. Currently available datasets are (random, fp_ldos)" % args.dataset)
    exit(0)

hvd.allreduce(torch.tensor(0), name='barrier')
print("Rank: %d, Creating train sampler/loader" % hvd.rank())

# TRAINING DATA
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
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



#test_sampler = torch.utils.data.distributed.DistributedSampler(
#    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
#test_loader = torch.utils.data.DataLoader(
#    test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)


# Neural Network Class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mid_layers = 2
        self.hidden_dim = args.ldos_length

        self.fc1 = nn.Linear(args.fp_length, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, dens_length)
        self.fc4 = nn.Linear(300, args.ldos_length * lstm_in_length)
        self.fc5 = nn.Linear(args.ldos_length * lstm_in_length, args.ldos_length)

        if (model_choice == 4):
            self.my_lstm = nn.LSTM(args.ldos_length, self.hidden_dim, lstm_in_length)
        elif (model_choice == 5):
            self.my_lstm = nn.LSTM(args.ldos_length, int(self.hidden_dim / 2), lstm_in_length, bidirectional=True)

    def init_hidden_train(self):
        

        #        h0 = torch.empty(lstm_in_length * 2, int(args.batch_size / hvd.size()), self.hidden_dim // 2)
#        c0 = torch.empty(lstm_in_length * 2, int(args.batch_size / hvd.size()), self.hidden_dim // 2)

        h0 = torch.empty(lstm_in_length * 2, args.batch_size, self.hidden_dim // 2)
        c0 = torch.empty(lstm_in_length * 2, args.batch_size, self.hidden_dim // 2)
        
        h0.zero_()
        c0.zero_()

        return (h0, c0) 
 
    def init_hidden_test(self):
        

        h0 = torch.empty(lstm_in_length * 2, args.test_batch_size, self.hidden_dim // 2)
        c0 = torch.empty(lstm_in_length * 2, args.test_batch_size, self.hidden_dim // 2)

        h0.zero_()
        c0.zero_()

        return (h0, c0) 
              

    def forward(self, x, hidden):

        # MODEL 4 and 5
        # LDOS prediction with LSTM (uni-directional or bi-directional)
        if (model_choice == 4 or model_choice == 5):
            self.batch_size = x.shape[0]
           
#            print("Forward BS: %d" % self.batch_size)

            x = F.relu(self.fc1(x))
            for i in range(self.mid_layers):
                x = F.relu(self.fc2(x))
            x = F.relu(self.fc4(x))
            x, hidden = self.my_lstm(x.view(lstm_in_length, self.batch_size, args.ldos_length), hidden)
            x = x[-1].view(self.batch_size, -1)

        # MODEL 1
        # Density prediction 
        elif (model_choice == 1):
            x = F.relu(self.fc1(x))
            for i in range(self.mid_layers):
                x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

        # MODEL 2
        # Density prediction, no activation
        elif (model_choice == 2):
            x = self.fc1(x)
            for i in range(self.mid_layers):
                x = self.fc2(x)
            x = self.fc3(x)

        # MODEL 3
        # LDOS prediction without LSTM
        elif (model_choice == 3):
            x = F.relu(self.fc1(x))
            for i in range(self.mid_layers):
                x = F.relu(self.fc2(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))

        return x, hidden


model = Net()

if (model_choice == 4 or model_choice == 5):
#    model.hidden, model.cell = model.init_hidden()
    model.train_hidden = model.init_hidden_train()
    model.test_hidden = model.init_hidden_test()

if args.cuda:
    # Move model to GPU.
    model.cuda()
    model.train_hidden = (model.train_hidden[0].cuda(), model.train_hidden[1].cuda())
    model.test_hidden = (model.test_hidden[0].cuda(), model.test_hidden[1].cuda())
#    model.cell = model.cell.cuda()

# Count number of network parameters
#num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("\nNum Params: %d " % num_params)
#exit(0);


#model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])



# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                      momentum=args.momentum, nesterov=True)

#optimizer = optim.Adam(model.parameters(), lr=args.lr * hvd.size())

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)


def metric_average(val, name):
#    tensor = torch.tensor(val)
#    tensor = val.clone().detach()
#    avg_tensor = hvd.allreduce(tensor, name=name)
#    return avg_tensor.item()

    return val

# Train FP-LDOS Model
def train(epoch):

    # Clear stored gradient
    model.zero_grad()

#    if (epoch == 0):
        # Initialize hidden state
#    model.hidden = model.init_hidden()
    
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)

    running_loss = 0.0

    hidden_n = model.train_hidden

    for batch_idx, (data, target) in enumerate(train_loader):
        
        model.zero_grad()

        # Move data and target to gpu
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        # Zero out gradients for the new batch
        optimizer.zero_grad()
        
        # RUN MODEL
        output, hidden_n = model(data, hidden_n)

#        print(data.shape)
#        print(output.shape)
#        print(target.shape)

        hidden_n = (hidden_n[0].detach(), hidden_n[1].detach())

        ldos_loss = F.mse_loss(output, target)
        ldos_loss.backward()
        optimizer.step()

#        model.train_hidden = hidden_n

        running_loss += ldos_loss.item()

        if (batch_idx % args.log_interval == 0 % args.log_interval and hvd.rank() == 0): 

            ldos_loss_val = metric_average(ldos_loss.item(), 'avg_ldos_loss')
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6E}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), ldos_loss_val))
            
#            args.writer.add_scalar('training loss rank%d' % hvd.rank(), \
#                running_loss / args.log_interval, \
#                epoch * len(train_loader) + batch_idx)
           
        if (batch_idx > 20):
            break


    model.train_hidden = hidden_n


    ldos_loss_val = ldos_loss.item()
    return ldos_loss_val

# Validate trained model for early stopping
def validate():
    model.eval()

    running_ldos_loss = 0.0

    hidden_n = model.test_hidden

    for batch_idx, (data, target) in enumerate(validation_loader):
       
#        print("Batch: %d" % batch_idx)

        # Move data and target to gpu
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        # RUN MODEL
        output, hidden_n = model(data, hidden_n)

        hidden_n = (hidden_n[0].detach(), hidden_n[1].detach())

        running_ldos_loss += F.mse_loss(output, target).item()
       
        if (batch_idx % args.log_interval == 0 % args.log_interval and hvd.rank() == 0):
            print("Validation batch_idx %d of %d" % (batch_idx, len(validation_loader)))

        if (batch_idx > 20):
            break

    ldos_loss_val = metric_average(running_ldos_loss, 'avg_ldos_loss')
    
    model.test_hidden = hidden_n

    return ldos_loss_val

# Test model, post training
def test():
    model.eval()

    running_ldos_loss = 0.0
    running_dens_loss = 0.0
    plot_ldos = True

    test_ldos = np.empty([args.grid_pts, args.ldos_length])

    data_idx = 0

    hidden_n = model.test_hidden

#    test_accuracy = 0.
    for batch_idx, (data, target) in enumerate(test_loader):
        
        # Move data and target to gpu
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # RUN MODEL
        output, hidden_n = model(data, hidden_n)

        hidden_n = (hidden_n[0].detach(), hidden_n[1].detach())

#        dens_output = ldos_calc.ldos_to_density(output, args.temp, args.gcc)       
#        dens_target = ldos_calc.ldos_to_density(target, args.temp, args.gcc)

#        bandE_output = ldos_calc.ldos_to_bandenergy(output, args.temp, args.gcc)
#        bandE_target = ldos_calc.ldos_to_bandenergy(target, args.temp, args.gcc)
#        bandE_true   = ldos_calc.get_bandenergy(args.temp, args.gcc)

        num_samples = output.shape[0] 

        if (args.cuda):
            test_ldos[data_idx:data_idx + num_samples, :] = output.cpu().detach().numpy()
        else:
            test_ldos[data_idx:data_idx + num_samples, :] = output.detach().numpy()

        data_idx += num_samples

        # sum up batch loss
        running_ldos_loss += F.mse_loss(output, target, size_average=None).item()
#        running_dens_loss += F.mse_loss(dens_output, dens_target, size_average=None).item()
#        bandE_loss += F.mse_loss(bandE_output, bandE_target, size_average=None).item()
#       bandE_true_loss += F.mse_loss(bandE_output, bandE_true, size_average=None).item()

        if (plot_ldos and hvd.rank() == 0):
            for i in range(args.tb_ldos_comparisons):
                for j in range(output.shape[1]):
                    args.writer.add_scalars('test ldos %d rank%d' % (i, hvd.rank()), \
                            {'LDOS-ML': output[i,j], 'True': target[i,j]}, j)


            plot_ldos = False

        if (batch_idx % args.log_interval == 0 % args.log_interval and hvd.rank() == 0):
            print("Test batch_idx %d of %d" % (batch_idx, len(test_loader)))

        if (batch_idx > 20):
            break
#    if (hvd.rank() == 0):
#        print("Done test predictions.\n\nCalculating Band Energies.\n")
#    predicted_bandE = ldos_calc.ldos_to_bandenergy(predicted_ldos, args.temp, args.gcc)
#    target_bandE = ldos_calc.ldos_to_bandenergy(target_ldos, args.temp, args.gcc)
#    qe_bandE = ldos_calc.get_bandenergy(args.temp, args.gcc, args.test_snapshot)


    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
#    ldos_loss /= len(test_sampler)
#    dens_loss /= len(test_sampler)

    # Horovod: average metric values across workers.
    ldos_loss_val = metric_average(running_ldos_loss, 'avg_ldos_loss')
#    dens_loss_val = metric_average(running_dens_loss, 'avg_dens_loss')
    
    dens_loss_val = running_dens_loss

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: \nAverage LDOS loss: %4.4E\nAverage Dens loss: %4.4E\n' % (ldos_loss_val, dens_loss_val))
        print('\nSaving LDOS predictions to %s\n' % args.model_dir + "/" + args.dataset + "_predictions")
        np.save(args.model_dir + "/" + args.dataset + "_predictions", test_ldos)

    model.test_hidden = hidden_n

    return ldos_loss_val


### TRAIN ####
tot_tic = timeit.default_timer()

train_time = 0; 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.optim_patience, mode="max", factor=0.5, verbose=True)

epoch_loss = 0.0
prev_validate_loss = 1e16
validate_loss = 0.0

curr_patience = 0

if (hvd.rank() == 0):
    print("\n\nBegin Training!\n\n")

for epoch in range(1, args.epochs + 1):

    tic = timeit.default_timer()
    epoch_loss = train(epoch)
    
    # Global barrier
    #hvd.allreduce(torch.tensor(0), name='barrier')  
    toc = timeit.default_timer()
    
    scheduler.step(epoch_loss)
    
    if (hvd.rank() == 0):
        print("\nEpoch %d of %d, Training time: %3.3f\n" % (epoch, args.epochs, toc - tic))
    train_time += toc - tic
    
    # Early Stopping 
    validate_loss = validate()

    if (validate_loss < prev_validate_loss * args.early_stopping):
        print("\nValidation loss has decreased from %4.6e to %4.6e\n" % (prev_validate_loss, validate_loss))
        prev_validate_loss = validate_loss
        curr_patience = 0
    else:
        print("\nValidation loss has NOT decreased enough! (from %4.6e to %4.6e) Patience at %d of %d\n" % \
            (prev_validate_loss, validate_loss, curr_patience + 1, args.early_patience))
        curr_patience += 1
        if (curr_patience >= args.early_patience):
            print("\n\nPatience has been reached! Final validation error %4.6e\n\n" % validate_loss)
            break;

if (hvd.rank() == 0):
    print("\n\nTraining Complete!\n\n")

tic = timeit.default_timer()
test_loss = test()

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
