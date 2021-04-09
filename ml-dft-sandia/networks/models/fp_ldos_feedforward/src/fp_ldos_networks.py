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
        self.fc_width_width = []
        for i in range(args.ff_mid_layers):
            self.fc_width_width.append(nn.Linear(args.ff_width, args.ff_width))

        self.fc_width_width = nn.ModuleList(self.fc_width_width)

        # Big Spiker
        self.fc_width_lstm_in = nn.Linear(args.ff_width, args.ldos_length * args.lstm_in_length)
        self.fc_lstm_in_ldos = nn.Linear(args.ldos_length * args.lstm_in_length, args.ldos_length)
        
        # Activation Functions
        self.activation = nn.LeakyReLU()
        self.final_activation = nn.LeakyReLU()
        #self.final_activation = nn.ReLU()

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
                x = self.activation(self.fc_width_width[i](x))
       
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
 
###-----------------------------------------------------------------------###

#
# FP -> LDOS, TRANSFORMER Network
#
# Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class FP_LDOS_TRANSFORMER_Net(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args 

        self.dropout = .2

        # must be divisor of fp_length
        self.num_heads = 7

#        input/hidden equal lengths
#        self.num_hidden = 91
#        self.num_tokens = 400

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.args.fp_length, self.dropout)

        encoder_layers = nn.TransformerEncoderLayer(self.args.fp_length, self.num_heads, self.args.fp_length, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.args.ff_mid_layers)
#        self.encoder = nn.Embedding(self.num_tokens, self.args.fp_length)

        self.decoder = nn.Linear(self.args.fp_length, self.args.ldos_length)

        self.init_weights()
        
    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def init_weights(self):
        initrange = 0.1
#        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


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

        if self.src_mask is None or self.src_mask.size(0) != x.size(0):
            device = x.device
            mask = self.generate_square_subsequent_mask(x.size(0)).to(device)
            self.src_mask = mask

#        x = self.encoder(x) * math.sqrt(self.args.fp_length)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        output = self.decoder(output)

        return (output, hidden)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Need to develop better form here.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        div_term2 = torch.exp(torch.arange(0, d_model - 1 , 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term2)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
