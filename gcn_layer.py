import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
class GCNLayer(nn.Module):

    def __init__(self,
                 g,
                 in_dim,
                 out_dim,
                 activation,
                 normalize=True,
                 batch_norm=True,
                 dropout=0,
                 residual=True,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.activation = activation
        self.normalize = normalize
        self.batch_norm = batch_norm
        self.residual = residual
        self.bias = bias
        
        if self.in_channels != self.out_channels:
            self.residual = False

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        
        self.batchnorm_h = nn.BatchNorm1d(self.out_channels)

        if self.normalize:
            gcn_norm = 'both'
        else:
            gcn_norm = 'none'
        self.conv = GraphConv(self.in_channels, self.out_channels, norm=gcn_norm, bias=self.bias)

        
    def forward(self, feature):
        h_in = feature
        if self.dropout is not None:
            h = self.dropout(h_in)
        h = self.conv(self.g, feature)

        if self.batch_norm:
            # combine all non-feature dimensions
            shape_orig = h.shape
            h = h.view(-1, h.shape[-1])
            h = self.batchnorm_h(h)
            h = h.view(*shape_orig)

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, batch_norm={}, dropout={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels,
                                             self.batch_norm, 
                                             self.dropout, 
                                             self.residual)


"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
"""
class GraphSAGELayer(nn.Module):

    def __init__(self,
                 g,
                 in_dim,
                 out_dim,
                 activation,
                 batch_norm=True,
                 residual=True,
                 bias=True):
        super(GraphSAGELayer, self).__init__()
        self.g = g
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.activation = activation
        self.aggregator = "mean"
        self.batch_norm = batch_norm
        self.residual = residual
        self.bias = bias
        
        if self.in_channels != self.out_channels:
            self.residual = False
        
        self.batchnorm_h = nn.BatchNorm1d(self.out_channels)

        self.conv = SAGEConv(self.in_channels, self.out_channels, self.aggregator, bias=self.bias)

        
    def forward(self, feature):
        h_in = feature   # to be used for residual connection
        h = self.conv(self.g, feature)

        if self.batch_norm:
            # combine all non-feature dimensions
            shape_orig = h.shape
            h = h.view(-1, h.shape[-1])
            h = self.batchnorm_h(h)
            h = h.view(*shape_orig)

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h # residual connection

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, batch_norm={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels,
                                             self.aggregator,
                                             self.batch_norm, self.residual)

