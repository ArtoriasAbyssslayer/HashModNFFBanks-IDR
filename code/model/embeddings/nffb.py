
"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    Then we make multiple level of encodings and get a multi-resolution embeddings and sum them up to

"""

import torch
import torch.nn as nn

from model.embeddings.fourier_encoding import FourierEncoding as FFenc
from model.embeddings.hash_encoder.hashgridencoder import MultiResolutionHashEncoderCUDA as MultiResHashGridEncoderCUDA
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




class FourierFilterBanks(nn.Module):
    def __init__(self, GridEncoderConfig, d_in, bound = 1.0, has_out = True):
        super().__init__()
        
        self.bound = bound
        
        ### Encoder Part
        ffenc_dims = GridEncoderConfig['n_levels'] 
        ffenc_dims = [d_in] + ffenc_dims
        self.n_ffenc_layers  = len(ffenc_dims) 

        features_per_level = GridEncoderConfig['max_points_per_level']
        base_resolution = GridEncoderConfig['base_resolution']
        per_level_scale = GridEncoderConfig['per_level_scale']
        
        
        assert self.n_ffenc_layers > 3, "The multiresolution Fourier Feature Encoding (Branch) should be greater than 3"
        grid_levels = int(self.n_ffenc_layers - 2)
        self.hashgrid_encoder = MultiResHashGridEncoderCUDA(input_dim=d_in, 
                                                            num_levels=grid_levels, 
                                                            level_dim=features_per_level, 
                                                            per_level_scale=per_level_scale, 
                                                            base_resolution=base_resolution, 
                                                            log2_hashmap_size=GridEncoderConfig['log2_hashmap_size'], 
                                                            desired_resolution=GridEncoderConfig['desired_resolution'])
        
        self.n_grid_levels = grid_levels
        print(f"Grid encoder levels: {grid_levels}")
        self.feat_dim = features_per_level
        
        
        # Create the FourierFeatureNetworks to map low-dim grid features to high-dim FourierGridFeatures
        ffn_list = []
        ffenc_kwargs = {
            'include_input': False,
            'input_dims': GridEncoderConfig['in_dim'],
            'max_freq_log2': GridEncoderConfig['log2_hashmap_size'],
            'num_freqs': GridEncoderConfig['n_levels'],
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos]
        }
        for i in range(grid_levels):
            ffn_list.append = FFenc(**ffenc_kwargs)