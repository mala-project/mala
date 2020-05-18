## networks for density prediction

import numpy as np
import torch
from functools import partial

from e3nn.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.non_linearities import GatedBlock
from e3nn.non_linearities import rescaled_act
from e3nn.non_linearities.rescaled_act import relu, sigmoid
from e3nn.radial import CosineBasisModel
from e3nn.radial import GaussianRadialModel



class Mixer(torch.nn.Module):
    def __init__(self, Op, Rs_in_s, Rs_out):
        super().__init__()
        self.ops = torch.nn.ModuleList([
            Op(Rs_in, Rs_out)
            for Rs_in in Rs_in_s
        ])

    def forward(self, *args, n_norm=1):
        # It simply sums the different outputs
        y = 0
        for m, x in zip(self.ops, args):
            y += m(*x, n_norm=n_norm)
        return y


class MixerNetwork(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out_list, max_radius=3.0, number_of_basis=3, radial_layers=3, basistype="Gaussian"):
        super().__init__()

        #sp = rescaled_act.Softplus(beta=5)
        #sp = rescaled_act.ShiftedSoftplus(beta=5)
        sp = torch.nn.Tanh()

        # the [0] is just to get first_layer in stripped form.
        # will not work for Rs_in with more than L=0
        first_layer = Rs_in[0]
        last_shared_layer = (2,1,1)

        representations = [first_layer, last_shared_layer]
        representations = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations]

        if (basistype == 'Gaussian'):
            rad_basis = GaussianRadialModel
        elif (basistype == 'Cosine'):
            rad_basis = CosineBasisModel
        else:
            print ("Only Gaussian and Cosine Radial basis are currently supported")

        RadialModel = partial(rad_basis, max_radius=max_radius,
                              number_of_basis=number_of_basis, h=100,
                              L=radial_layers, act=sp)

        K = partial(Kernel, RadialModel=RadialModel)
        C = partial(Convolution, K)
        M = partial(Mixer, C)  # wrap C to accept many input types

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, sp, sigmoid)
            conv = Convolution(K, Rs_in, act.Rs_in)
            return torch.nn.ModuleList([conv, act])

        self.layers = torch.nn.ModuleList([
            make_layer(Rs_layer_in,Rs_layer_out)
            for Rs_layer_in, Rs_layer_out in zip(representations[:-1], representations[1:])
        ])

        ## set up the split final layer
        m = []
        for rs in Rs_out_list:
            m.append(M([representations[-1], representations[-1]], rs))
        
        # final layer is indexed in order of atom type
        self.final_layer = torch.nn.ModuleList([
            m[i] for i in range(len(m))
        ])

    def forward(self, input, geometry, atom_type_map):
        output = input
        batch, N, _ = geometry.shape

        for conv, act in self.layers:
            output = conv(output, geometry, n_norm=N)
            output = act(output)

        ## split final layer
        geometry_list = []
        feature_list = []
        for i, item in enumerate(atom_type_map):
            geometry_list.append(geometry[0][item])
            feature_list.append(output[0][item])

        ## this is assuming that there are only two atom types!
        ## it should work, though for any arbitrary order of O and H in xyzfile!
        featuresO = feature_list[0].unsqueeze(0)
        featuresH = feature_list[1].unsqueeze(0)
        geometryO = geometry_list[0].unsqueeze(0)
        geometryH = geometry_list[1].unsqueeze(0)
        
        final_layer_output = []
        for i, layer in enumerate(self.final_layer):
            if (i == 0):
                final = layer((featuresO, geometryO, geometryO), (featuresH, geometryH, geometryO), n_norm = N)
            if (i == 1):
                final = layer((featuresO, geometryO, geometryH), (featuresH, geometryH, geometryH), n_norm = N) 
            final_layer_output.append(final)

        # return list of outputO and outputH
        output = final_layer_output

        return output



class SplitNetwork(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out_list, max_radius=3.0, number_of_basis=3, radial_layers=3, basistype="Gaussian"):
        super().__init__()

        #sp = rescaled_act.Softplus(beta=5)
        sp = rescaled_act.ShiftedSoftplus(beta=5)
        #sp = torch.nn.Tanh()

        # the [0] is just to get first_layer in stripped form.
        # will not work for Rs_in with more than L=0
        first_layer = Rs_in[0]
        #last_shared_layer = (2,1,1)
        last_shared_layer = (1,1,1,1,1,1)

        representations = [first_layer, last_shared_layer]
        representations = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations]

        if (basistype == 'Gaussian'):
            rad_basis = GaussianRadialModel
        elif (basistype == 'Cosine'):
            rad_basis = CosineBasisModel
        else:
            print ("Only Gaussian and Cosine Radial basis are currently supported")

        RadialModel = partial(rad_basis, max_radius=max_radius,
                              number_of_basis=number_of_basis, h=100,
                              L=radial_layers, act=sp)

        K = partial(Kernel, RadialModel=RadialModel)

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, sp, sigmoid)
            conv = Convolution(K, Rs_in, act.Rs_in)
            return torch.nn.ModuleList([conv, act])

        self.layers = torch.nn.ModuleList([
            make_layer(Rs_layer_in,Rs_layer_out)
            for Rs_layer_in, Rs_layer_out in zip(representations[:-1], representations[1:])
        ])

        ## set up the split final layer
        # final layer is indexed in order of atom type
        self.final_layer = torch.nn.ModuleList([
            Convolution(K, representations[-1], rs) for rs in Rs_out_list
        ])

    def forward(self, input, geometry, atom_type_map):
        output = input
        batch, N, _ = geometry.shape

        for conv, act in self.layers:
            output = conv(output, geometry, n_norm=N)
            output = act(output)

        ## split final layer
        geometry_list = []
        feature_list = []
        for i, item in enumerate(atom_type_map):
            geometry_list.append(geometry[0][item])
            feature_list.append(output[0][item])

        ## this is assuming that there are only two atom types!
        ## it should work, though for any arbitrary order of O and H in xyzfile!
        featuresO = feature_list[0].unsqueeze(0)
        featuresH = feature_list[1].unsqueeze(0)
        geometryO = geometry_list[0].unsqueeze(0)
        geometryH = geometry_list[1].unsqueeze(0)
        
        final_layer_output = []
        for i, layer in enumerate(self.final_layer):
            if (i == 0):
                final = layer(featuresO, geometryO)
                #final = layer((featuresO, geometryO, geometryO), (featuresH, geometryH, geometryO), n_norm = N)
            if (i == 1):
                final = layer(featuresH, geometryH)
                #final = layer((featuresO, geometryO, geometryH), (featuresH, geometryH, geometryH), n_norm = N) 
            final_layer_output.append(final)

        # return list of outputO and outputH
        output = final_layer_output

        return output