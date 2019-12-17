from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

import timeit
import numpy as np 


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--train-test-ratio', type=float, default=.80, metavar='R',
                    help='pecentage of training data to use (default: .80)')
parser.add_argument('--nxyz', type=int, default=20, metavar='N',
                    help='num elements along x,y,z dims (default: 20)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--model', type=int, default=1, metavar='N',
                    help='model choice (default: 1)')
parser.add_argument('--num-threads', type=int, default=32, metavar='N',
                    help='number of  (default: 32)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Horovod: initialize library.
hvd.init()
torch.manual_seed(args.seed)

if (args.batch_size < hvd.size()):
    print("Changing batch_size from %d to %d (number of ranks)" % (args.batch_size, hvd.size()))
    args.batch_size = hvd.size()

args.test_batch_size = args.batch_size

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)


# Horovod: limit # of CPU threads to be used per worker.

if (hvd.rank() == 0):
    print("Running with %d threads" % (args.num_threads))

torch.set_num_threads(args.num_threads)

# num_workers for multiprocessed data loading
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}


# Choose a Model

# Model 1: Density estimation with activations
# Model 2: Density estimation without activations
# Model 3: LDOS estimation without LSTM
# Model 4: LDOS estimation with LSTM
# Model 5: LDOS estimation with bidirectional LSTM

#model_choice = 4;

dataset = "fp_ldos"

temp = "300K"
gcc = "5.0"

training_path = "/ascldap/users/johelli/Code/mlmm/mlmm-ldrd-data/networks/training_data"

if (args.model != 0):
    model_choice = args.model

# Model params

#grid_pts = 200 ** 3
grid_pts = args.nxyz ** 3 * 4
test_pts = args.nxyz ** 3 * 4

bis_length = 80
ldos_length = 180
dens_length = 1
lstm_in_length = 10

if(hvd.rank() == 0):
    print("Running with %d rank(s)" % hvd.size())
    if (model_choice == 1):
        print("Building Density estimation model with activations")
    elif (model_choice == 2):
        print("Building Density estimation model without activations")
    elif (model_choice == 3):
        print("Building LDOS estimation model without LSTM")
    elif (model_choice == 4):
        print("Building LDOS estimation model with LSTM")
    elif (model_choice == 5):
        print("Building LDOS estimation model with bidirectional LSTM")
    else:
        print("Error in model choice");
        exit();

# Set up model outputs
if (dataset == "random"):
    if (model_choice == 1):
        train_np_y = torch.tensor(np.random.rand(grid_pts, dens_length), dtype=torch.float32)
        test_np_y = torch.tensor(np.random.rand(test_pts, dens_length), dtype=torch.float32)
    elif (model_choice == 2):
        train_np_y = torch.tensor(np.random.rand(grid_pts, dens_length), dtype=torch.float32)
        test_np_y = torch.tensor(np.random.rand(test_pts, dens_length), dtype=torch.float32)
    elif (model_choice == 3):
        train_np_y = torch.tensor(np.random.rand(grid_pts, ldos_length), dtype=torch.float32)
        test_np_y = torch.tensor(np.random.rand(test_pts, ldos_length), dtype=torch.float32)
    elif (model_choice == 4):
        train_np_y = torch.tensor(np.random.rand(grid_pts, ldos_length), dtype=torch.float32)
        test_np_y = torch.tensor(np.random.rand(test_pts, ldos_length), dtype=torch.float32)
    elif (model_choice == 5):
        train_np_y = torch.tensor(np.random.rand(grid_pts, ldos_length), dtype=torch.float32)
        test_np_y = torch.tensor(np.random.rand(test_pts, ldos_length), dtype=torch.float32)
    else:
        print("Error in model choice");
        exit();

    train_np_x = torch.tensor(np.random.rand(grid_pts, bis_length), dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(train_np_x, train_np_y)

    test_np_x = torch.tensor(np.random.rand(test_pts, bis_length), dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(test_np_x, test_np_y)

elif (dataset == "fp_ldos"):
    if (model_choice != 3 and model_choice != 4 and model_choice != 5):
        print("Error in model choice with fp_ldos dataset");
        exit();

    print("Reading Fingerprint and LDOS dataset")

    fp_np = np.load(training_path + "/mat_configs/%s/%sgcc/Al.fingerprint.npy" % (temp, gcc))
    ldos_np = np.load(training_path + "/ldos_data/ldos_%s_%sgcc_200x200x200grid_128elvls.npy" % (temp, gcc))

    fp_shape = fp_np.shape
    ldos_shape = ldos_np.shape

#    print(fp_shape)
#    print(ldos_shape)


    fp_pts = fp_shape[0] * fp_shape[1] * fp_shape[2]
    ldos_pts = ldos_shape[0] * ldos_shape[1] * ldos_shape[2]

    if (fp_pts != ldos_pts):
        print("Error in fp_pts %d and ldos_pts %d" % (fp_pts, ldos_pts));
        exit();

    grid_pts = fp_pts

    train_pts = int(grid_pts * args.train_test_ratio)
    test_pts = grid_pts - train_pts

    # Vector lengths
    bis_length = fp_shape[3]
    ldos_length = ldos_shape[3]
   
    if (hvd.rank() == 0):
        print("Grid_pts %d" % grid_pts)
        print("Train_pts %d, Test pts %d" % (train_pts, test_pts))
        print("Fingerprint vector length: %d" % bis_length)
        print("LDOS vector length: %d" % ldos_length)

    fp_np = fp_np.reshape([grid_pts, bis_length])
    ldos_np = ldos_np.reshape([grid_pts, ldos_length])

    # Row scaling of features 
    for row in range(bis_length):

        #meanv = np.mean(fp_np[row, :])
        maxv  = np.max(fp_np[row, :])
        minv  = np.min(fp_np[row, :])

        fp_np[row, :] = (fp_np[row, :] - minv) / (maxv - minv)
        #fp_np[row, :] = fp_np[row, :] / maxv

        if (hvd.rank() == 0):
            print("FP Row: %g, Min: %g, Avg: %g, Max: %g" % (row, np.min(fp_np[row, :]), np.mean(fp_np[row, :]), np.max(fp_np[row, :])))

    # Row scaling of outputs
    for row in range(ldos_length):

        #meanv = np.mean(fp_np[row, :])
        maxv  = np.max(ldos_np[row, :])
        minv  = np.min(ldos_np[row, :])

        ldos_np[row, :] = (ldos_np[row, :] - minv) / (maxv - minv)
        #fp_np[row, :] = fp_np[row, :] / maxv

        if (hvd.rank() == 0):
            print("LDOS Row: %g, Min: %g, Avg: %g, Max: %g" % (row, np.min(ldos_np[row, :]), np.mean(ldos_np[row, :]), np.max(ldos_np[row, :])))

    fp_torch = torch.tensor(fp_np, dtype=torch.float32)
    ldos_torch = torch.tensor(ldos_np, dtype=torch.float32)

    fp_ldos_dataset = torch.utils.data.TensorDataset(fp_torch, ldos_torch)

    train_dataset, test_dataset = torch.utils.data.random_split(fp_ldos_dataset, [train_pts, test_pts])



#exit()

# TRAINING DATA
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)


# TESTING DATA
# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)


#print("\n\nSucess!\n\n", flush=True)
#exit()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mid_layers = 2
        self.hidden_dim = ldos_length

        self.fc1 = nn.Linear(bis_length, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, dens_length)
        self.fc4 = nn.Linear(300, ldos_length * lstm_in_length)
        self.fc5 = nn.Linear(ldos_length * lstm_in_length, ldos_length)

        if (model_choice == 4):
            self.my_lstm = nn.LSTM(ldos_length, self.hidden_dim, lstm_in_length)
        elif (model_choice == 5):
            self.my_lstm = nn.LSTM(ldos_length, int(self.hidden_dim / 2), lstm_in_length, bidirectional=True)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(lstm_in_length, int(args.batch_size / hvd.size()), self.hidden_dim),
                torch.zeros(lstm_in_length, int(args.batch_size / hvd.size()), self.hidden_dim))

    def forward(self, x):
        self.batch_size = x.shape[0]
        # MODEL 1
        # Density prediction 
        if (model_choice == 1):
            x = F.relu(self.fc1(x))
            for i in range(self.mid_layers):
                x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

        # MODEL 2
        # Density prediction, no activation
        if (model_choice == 2):
            x = self.fc1(x)
            for i in range(self.mid_layers):
                x = self.fc2(x)
            x = self.fc3(x)

        # MODEL 3
        # LDOS prediction without LSTM
        if (model_choice == 3):
            x = F.relu(self.fc1(x))
            for i in range(self.mid_layers):
                x = F.relu(self.fc2(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))

        # MODEL 4 and 5
        # LDOS prediction with LSTM (uni-directional or bi-directional)
        if (model_choice == 4 or model_choice == 5):
            x = F.relu(self.fc1(x))
            for i in range(self.mid_layers):
                x = F.relu(self.fc2(x))
            x = F.relu(self.fc4(x))
            x, self.hidden = self.my_lstm(x.view(lstm_in_length, self.batch_size, ldos_length))
            x = x[-1].view(self.batch_size, -1)

        return x


model = Net()

model.hidden = model.init_hidden()

if args.cuda:
    # Move model to GPU.
    model.cuda()

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


def train(epoch):

    # Clear stored gradient
    model.zero_grad()

#    if (epoch == 0):
        # Initialize hidden state
#    model.hidden = model.init_hidden()
    
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        
        model.zero_grad()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

#        print(data.shape)
#        print(output.shape)
#        print(target.shape)

#        loss = F.nll_loss(output, target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.mse_loss(output, target, size_average=False).item()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))

tot_tic = timeit.default_timer()

train_time = 0; 


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

for epoch in range(1, args.epochs + 1):

    tic = timeit.default_timer()
    train(epoch)
    
    # Global barrier
    hvd.allreduce(torch.tensor(0), name='barrier')  
    toc = timeit.default_timer()
    
    if (hvd.rank() == 0):
        print("Epoch %d, Training time: %3.3f " % (epoch, toc - tic))
    train_time += toc - tic

    scheduler.step()

tic = timeit.default_timer()
test()

# Global barrier
hvd.allreduce(torch.tensor(0), name='barrier') 
toc = timeit.default_timer()
    
if (hvd.rank() == 0):
    print("Epoch %d, Testing time: %3.3f " % (epoch, toc - tic))


tot_toc = timeit.default_timer()

if (hvd.rank() == 0):
    print("Total train time %4.4f, Total test time %4.4f" % (train_time, (tot_toc - tot_tic) - train_time))
