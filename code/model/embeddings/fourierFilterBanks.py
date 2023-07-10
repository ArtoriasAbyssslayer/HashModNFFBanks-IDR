import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from model.embeddings.fourier_encoding import FourierEncoding as FourierFeatures
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
from model.embeddings.tcunn_implementations import hashGridEncoderTcnn as MRHashGridEncTcnn
"""
    Filter banks MLP is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    Then we make multiple level of encodings and get a multi-resolution embeddings and sum them up to

"""
import torch
import torch.nn as nn
from typing import Optional, List


class FourierFilterBanks(nn.Module):
    """
        FourierFilterBanks Constructor
        Args:
            in_dim(int): Embedding input layer dimensionality
            num_outputs(int): Embedding output layer outputs
            layer_channels(List): List of integer defining the dimensionality of each layer


            n_levels(int): Number of levels for MultiResHashGridMLP
            // Optional Arguments for MultiResHashGridMLP
            max_points_per_level(int): Maximum number of points per level for MultiResHashGridMLP
            log2_hashmap_size(int): Hashmap size for MultiResHashGridMLP
            base_resolution(int): Base resolution for MultiResHashGridMLP
            desired_resolution(int): Desired resolution for MultiResHashGridMLP
            // Optional Parameters for Fourier Features MLPs
            a_vals(torch.Tensor): a values in the fourier feature trans defining the scaling coefficient of each sinusoidal component [Scale]
            b_vals(torch.Tensor): b values in the fourier feature trans defining the harmonic freq of each sinusoidal component [Freq]
    """
    def __init__(self,
                 include_input:bool,
                 num_inputs:int,
                 num_outputs:int,
                 n_levels:int,
                 max_points_per_level:int,
                 a_vals,
                 b_vals,
                 layer_channels:List[int]):

        super(FourierFilterBanks, self).__init__()
        self.include_input = include_input
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.n_levels = n_levels
        self.max_points_per_level = max_points_per_level
        self.a_vals = a_vals
        self.b_vals = b_vals
        self.layer_channels = layer_channels


        if layer_channels is None:
            layer_channels = [num_inputs] * (n_levels + 1)
        self.mlp_layers = nn.ModuleList([nn.Linear(layer_channels[i], layer_channels[i+1]) for i in range(n_levels)])
        self.out_layer = nn.Linear(layer_channels[-1], num_outputs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute Fourier features
        Foruier_Grid_features = []
        for i in range(self.n_levels):
            x = self.mlp_layers[i](input)
            grid_features = MultiResHashGridMLP(x, self.max_points_per_level)
            fourier_grid_features = FourierFeaturesMLP(grid_features.shape[-1], self.mlp_layers[i].shape[-1],a_vals=self.a_vals,b_vals=self.b_vals,layer_channels=self.layer_channels)
            Foruier_Grid_features.append(fourier_grid_features)
        IndermediateOutputs = [None] * self.n_levels
        # Sum Fourier features with L_i MLP outputs
        for i in range(self.n_levels):
            if i == 0:
                IndermediateOutputs[i] = Foruier_Grid_features[i] + self.mlp_layers[i](x)
            else:
                IndermediateOutputs[i] = Foruier_Grid_features[i] + self.mlp_layers[i](IndermediateOutputs[i-1])
        x = torch.cat(IndermediateOutputs, dim=-1)
        output = self.out_layer(x)
        if self.include_input:
            output_dim = output.shape[-1] + input.shape[-1]
            self.register_parameter("output_dim", torch.nn.Parameter(torch.tensor(output_dim)))
            
        return output