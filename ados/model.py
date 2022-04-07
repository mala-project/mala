import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from gcn_layer import GCNLayer, GraphSAGELayer

BATCH_NORM = True
GC_BIAS = True


def disable_dropout(model):
    """Disable dropout on the model by setting eval mode on dropout layers"""
    for module in model.modules():
        if 'Dropout' in type(module).__name__:
            module.eval()
            

def enable_dropout(model):
    """Disable dropout on the model by setting train mode on dropout layers"""
    for module in model.modules():
        if 'Dropout' in type(module).__name__:
            module.train()


class CSGNN(nn.Module):

    def __init__(self, icosphere, radius_lvls, gconv_type, gconv_dims, gconv_depths, outblock_hidden_dims,
                       in_dim, out_dim, pool_type="none", conv_radial_layers=[], conv_radial_dims=[], k_radial=0,
                       pad=0, dropout=0, residual=True, gcn_norm=False, debug=False):
        super(CSGNN, self).__init__()
        self.icosphere = icosphere
        self.radius_lvls = radius_lvls
        self.max_level = icosphere.level
        self.gconv_dims = gconv_dims
        self.gconv_depths = gconv_depths
        self.gconv_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()
        self.pool_type = pool_type
        self.n_conv_blocks = len(self.gconv_depths)
        self.conv_radial_layers = conv_radial_layers
        self.conv_radial_dims = conv_radial_dims
        self.conv_radial_blocks = nn.ModuleList()
        self.activation = F.relu
        self.dropout = dropout
        self.k_radial = k_radial
        gconv_in_dim = in_dim
        R = self.radius_lvls
        for i in range(self.max_level):
            downsample_lvl = self.max_level-(i+1)
            gconv_out_dim = self.gconv_dims[i]
            gconv_h_dim = gconv_out_dim
            gconv_depth = gconv_depths[i]
            g = self.icosphere.graphs_by_level[self.max_level-i]
            dropout_gc = self.dropout
            if i == 0: # skip dropout 1st level
                dropout_gc = 0
            conv_block = ConvBlock(g, gconv_type, gconv_in_dim, gconv_h_dim, gconv_out_dim, gconv_depth, self.activation, gcn_norm, residual, dropout_gc, debug=debug)
            self.gconv_blocks.append(conv_block)
            gconv_in_dim = self.gconv_dims[i+1]
            
            # pooling and downsampling vertices
            pool_block = Pool(downsample_lvl, self.icosphere.base_pool_inds[downsample_lvl],
                                self.icosphere.rest_pool_inds[downsample_lvl], pool_type=self.pool_type)
            self.pool_blocks.append(pool_block)

            conv_radial_in_dim = self.gconv_dims[i]
            conv_radial_out_dim = self.conv_radial_dims[i]
            n_radial_convs = self.conv_radial_layers[i]
            conv_radial_block = ConvRadialBlock(conv_radial_in_dim, conv_radial_out_dim, n_radial_convs, k_radial, self.dropout, pad=pad, debug=debug)
            self.conv_radial_blocks.append(conv_radial_block)

        # base level convolutions
        idx_last = self.max_level
        g = self.icosphere.graphs_by_level[0]
        gconv_out_dim = self.gconv_dims[idx_last]
        gconv_h_dim = gconv_out_dim
        gconv_depth = gconv_depths[idx_last]
        conv_block = ConvBlock(g, gconv_type, gconv_in_dim, gconv_h_dim, gconv_out_dim, gconv_depth, self.activation, gcn_norm, residual, self.dropout, debug=debug)
        self.gconv_blocks.append(conv_block)
        last_dim = gconv_out_dim

        conv_radial_in_dim = self.gconv_dims[idx_last]
        conv_radial_out_dim = self.conv_radial_dims[idx_last]
        n_radial_convs = self.conv_radial_layers[idx_last]
        conv_radial_block = ConvRadialBlock(conv_radial_in_dim, conv_radial_out_dim, n_radial_convs, k_radial, self.dropout, pad=pad, debug=debug)
        self.conv_radial_blocks.append(conv_radial_block)
        last_dim = conv_radial_out_dim

        # final output
        outblock_in_dim = last_dim
        self.output = OutputBlock(outblock_in_dim, outblock_hidden_dims, out_dim, self.dropout, debug=debug)

    def forward(self, x):
        """
        Input: [n_batch, n_radius, n_mesh, n_feats]
        """

        is_eval = not self.training
        n_batch, n_radius, n_mesh, n_features = x.size()
        x = x.permute((2, 0, 1, 3)) # [n_mesh, n_batch, n_radius, n_features]

        # convolutions up to base level
        for i in range(self.max_level):
            x = self.gconv_blocks[i](x)
            x = self.conv_radial_blocks[i](x, is_eval)
            x = self.pool_blocks[i](x)

        idx_last = self.max_level
        x = self.gconv_blocks[idx_last](x)
        # pool vertex dimension
        if self.pool_type == "mean":
            x = torch.mean(x, 0, keepdim=True)
        elif self.pool_type == "max":
            x = torch.max(x, 0, keepdim=True)[0]
        elif self.pool_type == "sum":
            x = torch.sum(x, 0, keepdim=True)

        x = self.conv_radial_blocks[idx_last](x, is_eval)
    
        # pool R > 1 dimension
        if self.pool_type == "mean":
            x = torch.mean(x, 2)
        elif self.pool_type == "max":
            x = torch.max(x, 2)[0]
        elif self.pool_type == "sum":
            x = torch.sum(x, 2)

        x = torch.squeeze(x, dim=0) # n_batch, n_features

        x = self.output(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self,
                 g,
                 gconv_type,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 activation,
                 gcn_norm,
                 residual,
                 dropout,
                 debug=False):
        """
        n_layers: # hidden layers
        """
        super(ConvBlock, self).__init__()
        self.debug = debug
        self.gconv_type = gconv_type
        g = dgl.from_networkx(g)
        if gconv_type == "gcn":
            g = dgl.add_self_loop(g)
        self.g = g.to(torch.device('cuda'))
        self.conv_layers = nn.ModuleList()
        self.in_feats = in_feats
        self.out_feats = out_feats
        curr_in_feats = in_feats
        curr_out_feats = None
        for i in range(n_layers):
            curr_out_feats = n_hidden
            if self.gconv_type == "gcn": 
                self.conv_layers.append(GCNLayer(self.g, curr_in_feats, curr_out_feats, activation, normalize=gcn_norm, batch_norm=BATCH_NORM, dropout=dropout, residual=residual, bias=GC_BIAS))
            elif self.gconv_type == "graphsage":
                self.conv_layers.append(GraphSAGELayer(self.g, curr_in_feats, curr_out_feats, activation, batch_norm=BATCH_NORM, residual=residual, bias=GC_BIAS))

            curr_in_feats = curr_out_feats
        
        # last convolution layer
        curr_out_feats = out_feats
        if self.gconv_type == "gcn":
            self.conv_layers.append(GCNLayer(self.g, curr_in_feats, curr_out_feats, activation, normalize=gcn_norm, batch_norm=BATCH_NORM, dropout=dropout, residual=residual, bias=GC_BIAS))
        elif self.gconv_type == "graphsage":
            self.conv_layers.append(GraphSAGELayer(self.g, curr_in_feats, curr_out_feats, activation, batch_norm=BATCH_NORM, residual=residual, bias=GC_BIAS))

    def forward(self, features):
        h = features
        for i, conv_layer in enumerate(self.conv_layers):
            h = conv_layer(h)

        return h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, conv_layers=\
                \n{})'.format(self.__class__.__name__,
                                             self.in_feats,
                                             self.out_feats,
                                             self.conv_layers)


class Pool(nn.Module):
    """
    Pool and downsample input to match specified refinement mesh level
    """
    def __init__(self, level, base_pool_idx, rest_pool_idx, pool_type=None):
        super(Pool, self).__init__()
        self.level = level
        self.base_pool_idx = base_pool_idx
        self.rest_pool_idx = rest_pool_idx
        self.pool_type = pool_type

    def forward(self, x):
        if self.pool_type in ["sum", "max", "mean"]:
            x = self.pool(x)
        else:
            x = self.downSample(x)
        return x

    def downSample(self, x):
        """
        Downsample without pooling
        """
        nv_prev = 10*(4**self.level)+2
        return x[:nv_prev, ...]

    def pool(self, x):
        """
        Pooling over either 6 or 7 vertices' features
        (+1 from self vertex)
        """
        n_vertices, n_batch, n_radius, n_feats = x.shape
        base_pool_idx = self.base_pool_idx.reshape(-1)
        out_base = x[base_pool_idx, ...]
        out_base = out_base.reshape((-1, 6, n_batch, n_radius, n_feats))
        if self.pool_type == "sum":
            pooled_base = torch.sum(out_base, 1)
        elif self.pool_type == "mean":
            pooled_base = torch.mean(out_base, 1)
        elif self.pool_type == "max":
            pooled_base = torch.max(out_base, 1)[0]

        # rest_pool may be None (from L1 -> L0)
        if self.rest_pool_idx is not None:
            rest_pool_idx = self.rest_pool_idx.reshape(-1)
            out_rest = x[rest_pool_idx, ...]
            out_rest = out_rest.reshape((-1, 7, n_batch, n_radius, n_feats))
            if self.pool_type == "sum":
                pooled_rest = torch.sum(out_rest, 1)
            elif self.pool_type == "mean":
                pooled_rest = torch.mean(out_rest, 1)
            elif self.pool_type == "max":
                pooled_rest = torch.max(out_rest, 1)[0]
            out = torch.cat((pooled_base, pooled_rest), dim=0)
        else:
            out = pooled_base

        #print("out shape: {}".format(out.shape))
        return out

    def __repr__(self):
        return '{}(type={}, level={})'.format(self.__class__.__name__,
                                             self.pool_type,
                                             self.level)
        

class ConvRadialBlock(nn.Module):
    """
    Input: [V x B x N x R x F]
    Output: [V x B x N x R_out x F_out]
    """

    def __init__(self, in_dim, out_dim, n_convs, kernel, dropout, pad=0, debug=False):
        super(ConvRadialBlock, self).__init__()
        self.debug = debug
        self.pad = pad
        self.kernel = kernel
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        curr_dim = in_dim
        next_dim = out_dim
        for i in range(n_convs):
            self.conv_layers.append(nn.Conv1d(curr_dim, next_dim, kernel, padding=pad))
            if self.debug:
                self.batch_norms.append(nn.BatchNorm1d(next_dim, track_running_stats=False))
            else:
                self.batch_norms.append(nn.BatchNorm1d(next_dim))
            curr_dim = next_dim
            next_dim = out_dim
        self.nonlinearity = F.relu

    def forward(self, x, is_eval=False):
        V, B, R_in, F_in = x.shape
        x = x.permute((0, 1, 3, 2)) # V, B, F_in, R_in
        x = x.reshape(V*B, F_in, R_in)
        for i in range(len(self.conv_layers)):
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.conv_layers[i](x)
            if is_eval: # enable/disable cudnn to handle batchnorm cudnn issue
                torch.backends.cudnn.enabled = False
            x = self.batch_norms[i](x)
            torch.backends.cudnn.enabled = True
            x = self.nonlinearity(x)

        _, F_out, R_out = x.shape
        x = x.reshape(V, B, F_out, R_out)
        x = x.permute((0, 1, 3, 2)) # V, B, R_out, F_out
        return x

    def __repr__(self):
        return '{}(pad={}, k={}, dropout={}, layers={})'.format(self.__class__.__name__,
                                             self.pad,
                                             self.kernel,
                                             self.dropout, 
                                             self.conv_layers)


class OutputBlock(nn.Module):
    """
    Input: [B x N x F]
    Output: [B x C]
    """

    def __init__(self, f, h_dims, out_dim, dropout, debug=False):
        """ f:         input filters
            h_dims:    hidden dimensions
            out_dim:   output dimension
        """

        super(OutputBlock, self).__init__()
        self.debug = debug
        self.f = f
        self.hidden_dims = h_dims
        self.out_dim = out_dim
        self.dense_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.nonlinearity = F.relu
        curr_dim = f
        next_dim = None
        # hidden_dims may be empty
        for h in self.hidden_dims:
            next_dim = h
            self.dense_layers.append(nn.Linear(curr_dim, next_dim)) 
            if self.debug:
                self.batch_norms.append(nn.BatchNorm1d(next_dim, track_running_stats=False))
            else:
                self.batch_norms.append(nn.BatchNorm1d(next_dim))
            curr_dim = next_dim

        # output layer
        next_dim = out_dim
        self.dense_layers.append(nn.Linear(curr_dim, next_dim))
        if self.debug:
            self.batch_norms.append(nn.BatchNorm1d(next_dim, track_running_stats=False))
        else:
            self.batch_norms.append(nn.BatchNorm1d(next_dim))

        self.conv1d = nn.Conv1d(1, 3, 3, padding=1) # in_dim, out_dim, kernel size

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None


    def forward(self, x):
        """
        Input: [B x F]
        """
        n_batch, n_feats = x.shape
        x = x.reshape(n_batch, n_feats)
        ind_layer_last = len(self.dense_layers)-1
        for i, dense_layer in enumerate(self.dense_layers):
            if self.dropout is not None:
                x = self.dropout(x)
            x = dense_layer(x)
            x = self.batch_norms[i](x)
            x = self.nonlinearity(x)

        # conv1d smoothing over output bins
        x = x.reshape(n_batch, 1, -1)
        x = self.conv1d(x)
        x = self.nonlinearity(x)
        x = torch.mean(x, axis=1)

        # atomic dos
        x = x.reshape(n_batch, -1)

        return x

    def __repr__(self):
        return '{}(layers=\n{}\n{}\n{}'.format(self.__class__.__name__,
                                             self.dense_layers,
                                             self.conv1d,
                                             self.dropout)
