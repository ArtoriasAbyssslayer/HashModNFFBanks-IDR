
"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    Then we make multiple level of encodings and get a multi-resolution embeddings and sum them up to

"""

import torch
import torch.nn as nn

from model.embeddings.frequency_enc import FourierFeature as FFenc
from model.embeddings.hash_encoder.hashgridencoder import MultiResolutionHashEncoderCUDA as MultiResHashGridCUDA
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
#from model.embeddings.tcunn_implementations.hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as MultiResHashGridTCNN
class FourierFilterBanks(nn.Module):

    def __init__(self, GridEncoderNetConfig,has_out,bound):
        super(FourierFilterBanks, self).__init__()
        self.bound = bound
        self.include_input = GridEncoderNetConfig['include_input']
        self.num_inputs = GridEncoderNetConfig['num_inputs']
        self.num_outputs = GridEncoderNetConfig['num_outputs']
        self.n_levels = GridEncoderNetConfig['n_levels']
        self.max_points_per_level = GridEncoderNetConfig['max_points_per_level']

        # Compute Fourier features
        self.mlp_layers = []
        for i in range(self.n_levels):
            self.mlp_layers.append(nn.Linear(self.num_inputs, self.num_outputs))
        self.mlp_layers = nn.Sequential(*self.mlp_layers)
        self.grid_levels = int(self.n_levels - 1)
        print(f"Grid encoder levels: {self.grid_levels}")
        
        self.grid_enc =  MultiResHashGridMLP(self.include_input, self.num_inputs,
                                                self.n_levels, self.max_points_per_level,
                                                GridEncoderNetConfig['log2_hashmap_size'],
                                                GridEncoderNetConfig['base_resolution'],
                                                GridEncoderNetConfig['desired_resolution'])
        
        # 128 neurons are used to speed up the process of encoding
        ffenc_dims = [self.num_inputs]+[128]*self.grid_levels
        self.ffenc_dims = ffenc_dims
        ff_enc_list = []
        for i in range(self.grid_levels):
            ffenc_layer = FFenc(channels=self.max_points_per_level*ffenc_dims[2+i], sigma = GridEncoderNetConfig['base_sigma']*GridEncoderNetConfig['exp_sigma']**i,input_dims=ffenc_dims[i+1],include_input=True)
            ff_enc_list.append(ffenc_layer)
        self.ff_enc = nn.Sequential(*ff_enc_list)

        
        """ The Low - Frequency MLP part """
        idr_dims = GridEncoderNetConfig['network_dims']
        self.n_ffenc_layers = len(idr_dims)
        assert self.n_ffenc_layers > 6, "The Implicit Network Branch should have more than 6 layers"
        for layer in range(0, self.n_ffenc_layers - 1):
            setattr(self, "ff_lin" + str(layer), nn.Linear(ffenc_dims[layer], ffenc_dims[layer + 1]))
        
        """ The ouput layers """
        if has_out:
            self.out_layer = nn.Linear(self.ffenc_dims[-1],self.num_outputs)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        IndermediateOutputs = []
        x = input / self.bound  # Bound the input between [-1,1]
        input = (input + self.bound) / (2 * self.bound)
        
        # Compute HashGrid and split it to it's multiresolution levels
        augmented_grid_x = self.grid_enc(input)
        grid_x = augmented_grid_x[..., x.shape[-1]:]
        grid_x = grid_x.view(-1, self.grid_level, self.feature_dims)
        grid_x = grid_x.permute(1, 0, 2)
        
        for layer in range(self.grid_level):
            x = self.mlp_layers[layer](grid_x[layer])
            x = self.Fo
            if layer > 0:
                x = grid_x[i] + x
                
                
        # Sum Fourier features with L_i MLP outputs
        for i in range(self.n_levels):
            x = self.mlp_layers[i](x)
            if i == 0:
                IndermediateOutputs.append(self.Fourier_Grid_features[i].embed(x))
            else:
                x_high = self.Fourier_Grid_features[i-1](self.Fourier_Grid_features[i=1].embed(x))
                IndermediateOutputs.append(x_high)

        x = torch.sum(torch.stack(IndermediateOutputs), dim=0)
        output = self.out_layer(x)
        if self.include_input:
            output_dim = output.shape[-1] + input.shape[-1]
            self.register_parameter("output_dim", torch.nn.Parameter(torch.tensor(output_dim)))

        return output




# class FourierFilterBanks(nn.Module):
#     def __init__(self, GridEncoderConfig,HashEncoderType, d_in, bound = 1.0, has_out = True):
#         super().__init__()
        
#         self.bound = bound
        
#         ### Encoder Part
#         ffenc_dims = GridEncoderConfig['n_levels'] 
#         ffenc_dims = [d_in] + ffenc_dims
#         self.n_ffenc_layers  = len(ffenc_dims) 

#         features_per_level = GridEncoderConfig['max_points_per_level']
#         base_resolution = GridEncoderConfig['base_resolution']
#         per_level_scale = GridEncoderConfig['per_level_scale']
        
        
#         assert self.n_ffenc_layers > 3, "The multiresolution Fourier Feature Encoding (Branch) should be greater than 3"
#         grid_levels = int(self.n_ffenc_layers - 2)
#         if HashEncoderType == 'HashGridCUDA':
#             self.hashgrid_encoder = MultiResHashGridEncoderCUDA(input_dim=d_in, 
#                                                             num_levels=grid_levels, 
#                                                             level_dim=features_per_level, 
#                                                             per_level_scale=per_level_scale, 
#                                                             base_resolution=base_resolution, 
#                                                             log2_hashmap_size=GridEncoderConfig['log2_hashmap_size'], 
#                                                             desired_resolution=GridEncoderConfig['desired_resolution'])
#         elif HashEncoderType == 'HashGrid':
#             self.hashgrid_encoder = MultiResHashGridMLP(include_input=GridEncoderConfig['include_input'],
#                                                         in_dim=d_in
#                                                         )
#         elif HashEncoderType == 'HashGridTCNN':
#             self.hashgrid_encoder = MultiResHashGridTCNN(include_input=GridEncoderConfig['include_input'],
#                                                         in_dim=d_in
#                                                         )
#         else:
#             raise NotImplementedError(f"HashEncoderType {HashEncoderType} is not implemented")
        
#         self.n_grid_levels = grid_levels
#         print(f"Grid encoder levels: {grid_levels}")
#         self.feat_dim = features_per_level
        
        
#         # Create the FourierFeatureNetworks to map low-dim grid features to high-dim FourierGridFeatures
#         ffn_list = []
#         ffenc_kwargs = {
#             'include_input': False,
#             'input_dims': GridEncoderConfig['in_dim'],
#             'max_freq_log2': GridEncoderConfig['log2_hashmap_size'],
#             'num_freqs': GridEncoderConfig['n_levels'],
#             'log_sampling': True,
#             'periodic_fns': [torch.sin, torch.cos]
#         }
#         for i in range(grid_levels):
#             ffn_list.append = FFenc(**ffenc_kwargs)
#             # this should apply to every hashgrid resolution level the fourier feature encoding 
#             # and then we can sum them up
#             grid_features = self.hashgrid_encoder.grid_features
#             fourier_grid_features[i] = ffn_list[i].embed(grid_features[i])