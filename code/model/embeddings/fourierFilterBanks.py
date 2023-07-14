import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embeddings.fourier_encoding import FourierEncoding as FourierFeatures
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
from model.embeddings.tcunn_implementations import hashGridEncoderTcnn as HashGridTCNN
        
"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    Then we make multiple level of encodings and get a multi-resolution embeddings and sum them up to

"""
class FourierFilterBanks(nn.Module):

    def __init__(self,
                 GridEncoderNetConfig,
                 bound):

        super(FourierFilterBanks, self).__init__()
        self.bound = bound
        self.include_input = GridEncoderNetConfig['include_input']
        num_inputs = GridEncoderNetConfig['num_inputs']
        num_outputs = GridEncoderNetConfig['num_outputs']
        n_levels = GridEncoderNetConfig['n_levels']
        self.max_points_per_level = GridEncoderNetConfig['max_points_per_level']


        if layer_channels is None:
            layer_channels = [num_inputs] * (n_levels + 1)
        self.mlp_layers = nn.ModuleList([nn.Linear(layer_channels[i], layer_channels[i+1]) for i in range(n_levels)])
        self.out_layer = nn.Linear(layer_channels[-1], num_outputs)
        # Compute Fourier features
        Foruier_Grid_features = []
        for i in range(self.n_levels):
            x = self.mlp_layers[i](input)
            grid_features = HashGridTCNN(GridEncoderNetConfig)
            grid_features = MultiResHashGridMLP(x, self.max_points_per_level)
            grid_ff = FourierFeatures(grid_features.shape[-1], self.mlp_layers[i].shape[-1])
            Foruier_Grid_features.append(grid_ff)
        self.Fourier_Grid_features = Foruier_Grid_features
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        IndermediateOutputs = []
        x = input / self.bound # Bound the input between [-1,1]
        # Sum Fourier features with L_i MLP outputs
        for i in range(self.n_levels):
            if i == 0:
                x = self.mlp_layers[i](x)
                IndermediateOutputs[i] = self.Foruier_Grid_features[i].embed(x)
            else:
                x = self.mlp_layers[i](x)
                x_high = self.Foruier_Grid_features[i].embed(x)
                IndermediateOutputs[i] = self.Foruier_Grid_features[i].embed(x_high)
        x = sum(IndermediateOutputs)
        output = self.out_layer(x)
        if self.include_input:
            output_dim = output.shape[-1] + input.shape[-1]
            self.register_parameter("output_dim", torch.nn.Parameter(torch.tensor(output_dim)))
            
        return output