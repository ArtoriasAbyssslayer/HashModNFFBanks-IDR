import torch
import torch.nn as nn
import tinycudann as tcnn 
import math 

from model.embeddings.fourier_encoding import FourierEncoding
from hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as HashEncoderTcnn





class FFB_encoder(nn.Module):
    def __init__(self,
                 in_dim:int,
                 include_input:bool,
                 feature_dims,
                 base_resolution,
                 per_level_scale,
                 n_levels,
                 multires,
                 base_sigma,
                 exp_sigma,
                 bound
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.mlp_network_config = mlp_network_config
        self.encoding_config = encoding_config
        self.bound = bound 
        self.n_levels = n_levels
        self.feature_dims = feature_dims
        
        # TCNN encoder init
        self.FourierFeatureEncoding = FourierEncoding()
        sin_dims = sin_dims + [in_dim]
        self.num_sin_layers = len(sin_dims)
        assert self.num_sin_layers > 3, "The layer number (SIREN branch) shoudl be greater than 3 "
        
        grid_level = int(self.num_sin_layers - 2)        
        self.grid_encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.feature_dims,
                "log2_hasmap_size": multires-1,
                "per_level_scale": per_level_scale,
            }
        )
        self.grid_level = grid_level
        print(f"Grid encoder levels: {self.grid_level}")
        
        # Create the FourierFeatureEncoding for low-dim grid feats to map
        # high-dim SIREN feats
        
        fffn_list = []
        self.ffn_sigma_list = []
        for i in range(grid_level):
            ffn_A = torch.randn((feature_dims,sin_dims[2+i]),requires_grad=True) # * base_sigma * exp_sigma ** i
            ffn_list.append(ffn_A)
            self.ffn           
            
            
            
        
        
    def forward(self, in_pos):
        """
        IN_pos - Positional Encoding ~ Fourier Feature ENcoding 
        in_pos: [N, 3], in [-bound, bound]

        in_pos (for grid features) should always be located in [0.0, 1.0]
        x (for SIREN branch) should always be located in [-1.0, 1.0]
        """

        x = in_pos / self.bound								# to [-1, 1]
        in_pos = (in_pos + self.bound) / (2 * self.bound) 	# to [0, 1]

        grid_x = self.grid_encoder(in_pos)
        grid_x = grid_x.view(-1, self.grid_level, self.feat_dim)
        grid_x = grid_x.permute(1, 0, 2)

        ffn_A_list = []
        for i in range(self.grid_level):
            ffn_A_list.append(self.ffn_A[i] * self.ffn_sigma_list[i])
        ffn_A = torch.stack(ffn_A_list, dim=0)

        grid_x = torch.bmm(grid_x, 2 * math.pi * ffn_A)
        grid_x = torch.sin(grid_x)

        x_out = torch.zeros(x.shape[0], self.out_dim, device=in_pos.device)

        ### Grid encoding
        for layer in range(0, self.num_sin_layers - 1):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            x = sin_lin(x)
            x = self.sin_activation(x)

            if layer > 0:
                x = grid_x[layer-1] + x

                sin_lin_high = getattr(self, "sin_lin_high" + str(layer-1))
                x_high = sin_lin_high(x)
                x_high = self.sin_activation_high(x_high)

                x_out = x_out + x_high

        x = x_out

        return x