
"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    Then we make multiple level of encodings and get a multi-resolution embeddings and sum them up to

"""

import torch
import torch.nn as nn
from model.embeddings.fourier_encoding import FourierEncoding as FFenc
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP

class FourierFilterBanks(nn.Module):

    def __init__(self, GridEncoderNetConfig, bound):
        super(FourierFilterBanks, self).__init__()
        self.bound = bound
        self.include_input = GridEncoderNetConfig['include_input']
        self.num_inputs = GridEncoderNetConfig['num_inputs']
        self.num_outputs = GridEncoderNetConfig['num_outputs']
        self.n_levels = GridEncoderNetConfig['n_levels']
        self.max_points_per_level = GridEncoderNetConfig['max_points_per_level']

        # Compute Fourier features
        self.mlp_layers = nn.ModuleList()
        for i in range(self.n_levels):
            self.mlp_layers.append(nn.Linear(self.num_inputs, self.num_outputs))

        Fourier_Grid_features = []
        for i in range(self.n_levels):
            grid_features = MultiResHashGridMLP(self.include_input, self.num_inputs,
                                                self.n_levels, self.max_points_per_level,
                                                GridEncoderNetConfig['log2_hashmap_size'],
                                                GridEncoderNetConfig['base_resolution'],
                                                GridEncoderNetConfig['desired_resolution'])
            grid_ff = FFenc(grid_features.shape[-1], self.mlp_layers[i].shape[-1])
            Fourier_Grid_features.append(grid_ff)
        self.Fourier_Grid_features = Fourier_Grid_features

        self.out_layer = nn.Linear(self.num_outputs, self.num_outputs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        IndermediateOutputs = []
        x = input / self.bound  # Bound the input between [-1,1]

        # Sum Fourier features with L_i MLP outputs
        for i in range(self.n_levels):
            x = self.mlp_layers[i](x)
            if i == 0:
                IndermediateOutputs.append(self.Fourier_Grid_features[i].embed(x))
            else:
                x_high = self.Fourier_Grid_features[i].embed(x)
                IndermediateOutputs.append(self.Fourier_Grid_features[i].embed(x_high))

        x = torch.sum(torch.stack(IndermediateOutputs), dim=0)
        output = self.out_layer(x)
        if self.include_input:
            output_dim = output.shape[-1] + input.shape[-1]
            self.register_parameter("output_dim", torch.nn.Parameter(torch.tensor(output_dim)))

        return output
