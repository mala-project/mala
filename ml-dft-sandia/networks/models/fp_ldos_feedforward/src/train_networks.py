# FP LDOS, Networks

import os, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import horovod.torch as hvd

sys.path.append("../utils/")

#import ldos_calc

###-----------------------------------------------------------------------###

class Net_Trainer():
    def __init__(self, args):
        self.args = args

    def set_model(self, model):
        self.model = model
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return

    def set_data_samplers(self, train_sampler, validation_sampler, test_sampler):
        self.train_sampler = train_sampler
        self.validation_sampler = validation_sampler
        self.test_sample = test_sampler
        return

    def set_data_loaders(self, train_loader, validation_loader, test_loader):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        return


###-----------------------------------------------------------------------###

    def metric_average(self, val, name):
        #tensor = torch.tensor(val)
    #    tensor = val.clone().detach()
        #avg_tensor = hvd.allreduce(tensor, name=name)
        #return avg_tensor.item()

        return val

###-----------------------------------------------------------------------###

    # Train FP-LDOS Model
    def train(self, epoch):

        # Clear stored gradient
        self.model.zero_grad()
        
        self.model.train()
        # Horovod: set epoch to sampler for shuffling.
        self.train_sampler.set_epoch(epoch)

        running_loss = 0.0

        hidden_n = self.model.train_hidden

        for batch_idx, (data, target) in enumerate(self.train_loader):
            
            self.model.zero_grad()

            # Move data and target to gpu
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            
            # Zero out gradients for the new batch
            self.optimizer.zero_grad()

#            if (batch_idx % self.args.log_interval == 0 % self.args.log_interval and hvd.rank() == 0):
#                old_hidden = hidden_n

            # RUN self.model
            output, hidden_n = self.model(data, hidden_n)

            hidden_n = (hidden_n[0].detach(), hidden_n[1].detach()) 


#            if (batch_idx % self.args.log_interval == 0 % self.args.log_interval and hvd.rank() == 0):
#                print("Hidden diff: [%4.4f, %4.4f]" % (np.sum(np.abs(hidden_n[0].data.cpu().numpy() - old_hidden[0].data.cpu().numpy())), \
#                                                       np.sum(np.abs(hidden_n[1].data.cpu().numpy() - old_hidden[1].data.cpu().numpy()))))
            

            ldos_loss = F.mse_loss(output, target)
            ldos_loss.backward()
          
            # Gradient Clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()

            running_loss += ldos_loss.item()

            if (batch_idx % self.args.log_interval == 0 % self.args.log_interval and hvd.rank() == 0): 

                ldos_loss_val = self.metric_average(ldos_loss.item(), 'avg_ldos_loss')
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6E}'.format(
                    epoch, batch_idx * len(data), len(self.train_sampler),
                    100. * batch_idx / len(self.train_loader), ldos_loss_val))
 
#                for name, param in self.model.named_parameters():
#                    if param.requires_grad:
#                        print(batch_idx, name, param.data)
                                
#                self.args.writer.add_scalar('training loss rank%d' % hvd.rank(), \
#                    running_loss / self.args.log_interval, \
#                    epoch * len(self.train_loader) + batch_idx)
               
        self.model.train_hidden = hidden_n

        ldos_loss_val = ldos_loss.item()
        return ldos_loss_val


###-----------------------------------------------------------------------###

    # Validate trained model for early stopping
    def validate(self):
        self.model.eval()

        running_ldos_loss = 0.0

        hidden_n = self.model.test_hidden

        for batch_idx, (data, target) in enumerate(self.validation_loader):
           
    #        print("Batch: %d" % batch_idx)

            # Move data and target to gpu
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            
            # RUN self.model
            output, hidden_n = self.model(data, hidden_n)

            hidden_n = (hidden_n[0].detach(), hidden_n[1].detach())

            running_ldos_loss += F.mse_loss(output, target).item()
           
            if (batch_idx % self.args.log_interval == 0 % self.args.log_interval and hvd.rank() == 0):
                print("Validation batch_idx %d of %d" % (batch_idx, len(self.validation_loader)))

    #        if (batch_idx > 20):
    #            break

        ldos_loss_val = self.metric_average(running_ldos_loss, 'avg_ldos_loss')
        
        self.model.test_hidden = hidden_n

        return ldos_loss_val


###-----------------------------------------------------------------------###

    # Test model, post training
    def test(self):
        self.model.eval()

        running_ldos_loss = 0.0
        running_dens_loss = 0.0
        plot_ldos = True

        test_ldos = np.empty([self.args.grid_pts, self.args.ldos_length])

        data_idx = 0

        hidden_n = self.model.test_hidden

    #    test_accuracy = 0.
        for batch_idx, (data, target) in enumerate(self.test_loader):
            
            # Move data and target to gpu
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            # RUN self.model
            output, hidden_n = self.model(data, hidden_n)

            hidden_n = (hidden_n[0].detach(), hidden_n[1].detach())

    #        dens_output = ldos_calc.ldos_to_density(output, self.args.temp, self.args.gcc)       
    #        dens_target = ldos_calc.ldos_to_density(target, self.args.temp, self.args.gcc)

    #        bandE_output = ldos_calc.ldos_to_bandenergy(output, self.args.temp, self.args.gcc)
    #        bandE_target = ldos_calc.ldos_to_bandenergy(target, self.args.temp, self.args.gcc)
    #        bandE_true   = ldos_calc.get_bandenergy(self.args.temp, self.args.gcc)

            num_samples = output.shape[0] 

            if (self.args.cuda):
                test_ldos[data_idx:data_idx + num_samples, :] = output.cpu().detach().numpy()
            else:
                test_ldos[data_idx:data_idx + num_samples, :] = output.detach().numpy()

            data_idx += num_samples

            # sum up batch loss
            running_ldos_loss += F.mse_loss(output, target, size_average=None).item()
    #        running_dens_loss += F.mse_loss(dens_output, dens_target, size_average=None).item()
    #        bandE_loss += F.mse_loss(bandE_output, bandE_target, size_average=None).item()
    #       bandE_true_loss += F.mse_loss(bandE_output, bandE_true, size_average=None).item()

#            if (plot_ldos and hvd.rank() == 0):
#                for i in range(self.args.tb_ldos_comparisons):
#                    for j in range(output.shape[1]):
#                        self.args.writer.add_scalars('test ldos %d rank%d' % (i, hvd.rank()), \
#                                {'LDOS-ML': output[i,j], 'True': target[i,j]}, j)


#                plot_ldos = False

            if (batch_idx % self.args.log_interval == 0 % self.args.log_interval and hvd.rank() == 0):
                print("Test batch_idx %d of %d" % (batch_idx, len(self.test_loader)))

    #        if (batch_idx > 20):
    #            break

    #    if (hvd.rank() == 0):
    #        print("Done test predictions.\n\nCalculating Band Energies.\n")
    #    predicted_bandE = ldos_calc.ldos_to_bandenergy(predicted_ldos, self.args.temp, self.args.gcc)
    #    target_bandE = ldos_calc.ldos_to_bandenergy(target_ldos, self.args.temp, self.args.gcc)
    #    qe_bandE = ldos_calc.get_bandenergy(self.args.temp, self.args.gcc, self.args.test_snapshot)


        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
    #    ldos_loss /= len(test_sampler)
    #    dens_loss /= len(test_sampler)

        # Horovod: average metric values across workers.
        ldos_loss_val = self.metric_average(running_ldos_loss, 'avg_ldos_loss')
    #    dens_loss_val = self.metric_average(running_dens_loss, 'avg_dens_loss')
        
        dens_loss_val = running_dens_loss

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: \nAverage LDOS loss: %4.4E\nAverage Dens loss: %4.4E\n' % \
                    (ldos_loss_val, dens_loss_val))
            print('\nSaving LDOS predictions to %s\n' % self.args.model_dir + "/" + \
                    self.args.dataset + "_predictions")
            np.save(self.args.model_dir + "/" + self.args.dataset + "_predictions", test_ldos)

        self.model.test_hidden = hidden_n

        return ldos_loss_val


