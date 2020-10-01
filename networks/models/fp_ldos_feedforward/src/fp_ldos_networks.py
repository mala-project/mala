#
# FP -> LDOS, Networks
#

import os, sys
import numpy as np
import torch
import torch.nn as nn
import horovod.torch as hvd

import transformers

###-----------------------------------------------------------------------###

#
# FP -> LDOS, Feedforward Network
#
class FP_LDOS_FF_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args

        # First Layer
        self.fc_fp_width = nn.Linear(args.fp_length, args.ff_width)
        # Last Layer
        self.fc_width_ldos = nn.Linear(args.ff_width, args.ldos_length)
        
        
        # Stacked Autoencoders
        if (self.args.stacked_auto):
            self.fc_width_ae = nn.Linear(args.ff_width, args.ae_width)
            self.fc_ae_width = nn.Linear(args.ae_width, args.ff_width)
       
        # Deep Autoencoder
        if (self.args.deep_auto):
            self.fc_encode = []
            self.fc_decode = []

            encode_in = args.ff_width
            encode_out = int(encode_in * args.ae_factor)

            decode_in = encode_out
            decode_out = encode_in

            for i in range(args.ff_mid_layers):
                self.fc_encode.append(nn.Linear(encode_in, encode_out))
                self.fc_decode.append(nn.Linear(decode_in, decode_out))

                encode_in = encode_out
                encode_out = int(encode_in * args.ae_factor)

                decode_in = encode_out
                decode_out = encode_in

                self.fc_encode = nn.ModuleList(self.fc_encode)
                self.fc_decode = nn.ModuleList(self.fc_decode)

        # Flat Feedforward
        self.fc_width_width = nn.Linear(args.ff_width, args.ff_width) 

        # Big Spiker
        self.fc_width_lstm_in = nn.Linear(args.ff_width, args.ldos_length * args.lstm_in_length)
        self.fc_lstm_in_ldos = nn.Linear(args.ldos_length * args.lstm_in_length, args.ldos_length)
        
        # Activation Functions
        self.activation = nn.LeakyReLU()
        self.final_activation = nn.LeakyReLU()

    # Initialize hidden and cell states
    def init_hidden_train(self):
        h0 = torch.empty(1)
        c0 = torch.empty(1)
        
        h0.zero_()
        c0.zero_()

        return (h0, c0) 
 
    def init_hidden_test(self):
        h0 = torch.empty(1)
        c0 = torch.empty(1)
    
        h0.zero_()
        c0.zero_()

        return (h0, c0) 


    # Apply Network
    def forward(self, x, hidden = 0):

        self.batch_size = x.shape[0]

        x = self.activation(self.fc_fp_width(x))

        if (self.args.skip_connection):
            skip_x = x
        
        if (self.args.stacked_auto):
            for i in range(self.args.ff_mid_layers):
                x = self.activation(self.fc_width_ae(x))
                x = self.activation(self.fc_ae_width(x))

        elif (self.args.deep_auto):
            for i in range(self.args.ff_mid_layers):
                x = self.activation(self.fc_encode[i](x))
            
            for i in range(self.args.ff_mid_layers - 1, -1, -1):
                x = self.activation(self.fc_decode[i](x))

        else:
            for i in range(self.args.ff_mid_layers):
                x = self.activation(self.fc_width_width(x))
       
        if (self.args.skip_connection):
            x = x + skip_x

        x = self.activation(self.fc_width_lstm_in(x))

        return (self.final_activation(self.fc_lstm_in_ldos(x)), hidden)


###-----------------------------------------------------------------------###

#
# FP -> LDOS, LSTM Network
#
class FP_LDOS_LSTM_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.hidden_dim = args.ldos_length

        # First Layer
        self.fc_fp_width = nn.Linear(args.fp_length, args.ff_width)
        # Last Layer
        self.fc_width_ldos = nn.Linear(args.ff_width, args.ldos_length)
        
        
        # Stacked Autoencoders
        if (args.stacked_auto):
            self.fc_width_ae = nn.Linear(args.ff_width, args.ae_width)
            self.fc_ae_width = nn.Linear(args.ae_width, args.ff_width)
       
        # Deep Autoencoder
        if (args.deep_auto):
            self.fc_encode = []
            self.fc_decode = []

            encode_in = args.ff_width
            encode_out = int(encode_in * args.ae_factor)

            decode_in = encode_out
            decode_out = encode_in

            for i in range(args.ff_mid_layers):
                self.fc_encode.append(nn.Linear(encode_in, encode_out))
                self.fc_decode.append(nn.Linear(decode_in, decode_out))

                encode_in = encode_out
                encode_out = int(encode_in * args.ae_factor)

                decode_in = encode_out
                decode_out = encode_in

                self.fc_encode = nn.ModuleList(self.fc_encode)
                self.fc_decode = nn.ModuleList(self.fc_decode)

        # Flat Feedforward
        self.fc_width_width = nn.Linear(args.ff_width, args.ff_width) 

        # Big Spiker
        self.fc_width_lstm_in = nn.Linear(args.ff_width, args.ldos_length * args.lstm_in_length)

        # LSTM 
#        if (args.no_bidirection):
#            self.ldos_lstm = nn.LSTM(args.ldos_length, \
#                                     self.hidden_dim, \
#                                     args.lstm_in_length)
#        else:
#            self.ldos_lstm = nn.LSTM(args.ldos_length, \
#                                     int(self.hidden_dim / 2), \
#                                     args.lstm_in_length, \
#                                     bidirectional=True)
        if (args.gru):
            if (args.no_bidirection):
                self.ldos_lstm = nn.GRU(args.ldos_length, \
                                         self.hidden_dim, \
                                         args.lstm_in_length, \
                                         batch_first=True)
            else:
                self.ldos_lstm = nn.GRU(args.ldos_length, \
                                         int(self.hidden_dim / 2), \
                                         args.lstm_in_length, \
                                         batch_first=True, \
                                         bidirectional=True)


        else:
            if (args.no_bidirection):
                self.ldos_lstm = nn.LSTM(args.ldos_length, \
                                         self.hidden_dim, \
                                         args.lstm_in_length, \
                                         batch_first=True)
            else:
                self.ldos_lstm = nn.LSTM(args.ldos_length, \
                                         int(self.hidden_dim / 2), \
                                         args.lstm_in_length, \
                                         batch_first=True, \
                                         bidirectional=True)

        # Activation Functions
        self.activation = nn.LeakyReLU()
        self.final_activation = nn.LeakyReLU()

    # Initialize hidden and cell states
    def init_hidden_train(self):
                
        if (self.args.no_bidirection):
            h0 = torch.empty(self.args.lstm_in_length, \
                             self.args.batch_size, \
                             self.hidden_dim)
            c0 = torch.empty(self.args.lstm_in_length, \
                             self.args.batch_size, \
                             self.hidden_dim)
        else:
            h0 = torch.empty(self.args.lstm_in_length * 2, \
                             self.args.batch_size, \
                             self.hidden_dim // 2)
            c0 = torch.empty(self.args.lstm_in_length * 2, \
                             self.args.batch_size, \
                             self.hidden_dim // 2)

#        if (self.args.no_bidirection):
#            h0 = torch.empty(self.args.batch_size, \
        #                             self.args.lstm_in_length, \
        #                             self.hidden_dim)
#            c0 = torch.empty(self.args.batch_size, \
        #                             self.args.lstm_in_length, \
        #                             self.hidden_dim)
#        else:
#            h0 = torch.empty(self.args.batch_size, \
        #                             self.args.lstm_in_length * 2, \
        #                             self.hidden_dim // 2)
#            c0 = torch.empty(self.args.batch_size, \
#                             self.args.lstm_in_length * 2, \
#                             self.hidden_dim // 2)

        h0.zero_()
        c0.zero_()

        return (h0, c0) 
 
    def init_hidden_test(self):
       

        if (self.args.no_bidirection):
            h0 = torch.empty(self.args.lstm_in_length, \
                             self.args.test_batch_size, \
                             self.hidden_dim)
            c0 = torch.empty(self.args.lstm_in_length, \
                             self.args.test_batch_size, \
                             self.hidden_dim)
        else:
            h0 = torch.empty(self.args.lstm_in_length * 2, \
                             self.args.test_batch_size, \
                             self.hidden_dim // 2)
            c0 = torch.empty(self.args.lstm_in_length * 2, \
                             self.args.test_batch_size, \
                             self.hidden_dim // 2)


            #        if (self.args.no_bidirection):
#            h0 = torch.empty(self.args.test_batch_size, \
        #                             self.args.lstm_in_length, \
        #                             self.hidden_dim)
#            c0 = torch.empty(self.args.test_batch_size, \
        #                             self.args.lstm_in_length, \
        #                             self.hidden_dim)
#        else:
#            h0 = torch.empty(self.args.test_batch_size, \
        #                             self.args.lstm_in_length * 2, \
        #                             self.hidden_dim // 2)
#            c0 = torch.empty(self.args.test_batch_size, \
        #                             self.args.lstm_in_length * 2, \
        #                             self.hidden_dim // 2)
        
        h0.zero_()
        c0.zero_()

        return (h0, c0)  

    # Apply Network
    def forward(self, x, hidden = 0):

        self.batch_size = x.shape[0]

        if (self.args.no_hidden_state):
            hidden = (hidden[0].fill_(0.0), hidden[1].fill_(0.0))

        x = self.activation(self.fc_fp_width(x))

        if (self.args.skip_connection):
            skip_x = x
        
        if (self.args.stacked_auto):
            for i in range(self.args.ff_mid_layers):
                x = self.activation(self.fc_width_ae(x))
                x = self.activation(self.fc_ae_width(x))

        elif (self.args.deep_auto):
            for i in range(self.args.ff_mid_layers):
                x = self.activation(self.fc_encode[i](x))
            
            for i in range(self.args.ff_mid_layers - 1, -1, -1):
                x = self.activation(self.fc_decode[i](x))

        else:
            for i in range(self.args.ff_mid_layers):
                x = self.activation(self.fc_width_width(x))
       
        if (self.args.skip_connection):
            x = x + skip_x

        x = self.activation(self.fc_width_lstm_in(x))
 


#     if (self.args.no_bidirection):
#            x, hidden = self.ldos_lstm(x.view(self.args.lstm_in_length, \
        #                                              self.batch_size, \
        #                                              self.args.ldos_length), \
        #                                       hidden)
#        else:
#            x, hidden = self.ldos_lstm(x.view(self.args.lstm_in_length, \
        #                                              self.batch_size, \
        #                                              self.args.ldos_length), \
        #                                       hidden)





        if (self.args.no_bidirection):
            x, hidden = self.ldos_lstm(x.view(self.batch_size, \
                                              self.args.lstm_in_length, \
                                              self.args.ldos_length), \
                                       hidden)
        else:
            x, hidden = self.ldos_lstm(x.view(self.batch_size, \
                                              self.args.lstm_in_length, \
                                              self.args.ldos_length), \
                                       hidden)

#        print(x.shape)

#        x = x[-1].view(self.batch_size, -1)
        x = x[:, -1, :]
        return (self.final_activation(x), hidden)
 


class FP_LDOS_TRANSFORMER_Net(nn.Module):
    
    def init(self, args):
        self.args = args 
  

    def init_hidden_train(self):
        h0 = torch.empty(1)
        c0 = torch.empty(1)
        
        h0.zero_()
        c0.zero_()

        return (h0, c0) 


    def init_hidden_test(self):
        h0 = torch.empty(1)
        c0 = torch.empty(1)
        
        h0.zero_()
        c0.zero_()

        return (h0, c0) 


    def forward(self, x, hidden = 0):
        return (np.zeros(self.args.ldos_length), hidden)

    
