import torch
import torch.nn as nn
import tinycudann as tcnn 
import math 
from Sine import *
from model.embeddings.fourier_encoding import FourierEncoding as FFenc
from hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as HashEncoderTcnn





class FFB_encoder(nn.Module):
    def __init__(self,
                 in_dim:int,
                 include_input:bool,
                 feature_dims:int,
                 base_resolution,
                 num_outputs,
                 per_level_scale,
                 n_levels,
                 base_sigma,
                 exp_sigma,
                 grid_embed_std,
                 bound
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.bound = bound 
        self.n_levels = n_levels
        self.feature_dims = feature_dims
        self.multires = n_levels
        # HashTcnn Encoder init
        grid_level = int(self.num_sin_layers - 2)        
        self.grid_encoder = tcnn.Encoding(
            n_input_dims=in_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.feature_dims,
                "base_resolution": base_resolution,
                "base_sigma": base_sigma,
                "exp_sigma": exp_sigma,
                "log2_hasmap_size": n_levels-1,
                "per_level_scale": per_level_scale,
            }
        )
        self.grid_level = grid_level
        print(f"Grid encoder levels: {self.grid_level}")
        # FourierFeatureEncoding init
        self.num_frequencies = n_levels
        self.ff_enc = FFenc(include_input,in_dim,n_levels-1,self.num_frequencies,log_sampling=True,periodic_fns=[torch.sin,torch.cos])
        ffenc_dims = self.ff_enc.embeddings_dim + [in_dim]
        self.num_sin_layers = len(ffenc_dims)
        assert self.num_sin_layers > 3, "The layer number (SIREN branch) shoudl be greater than 3 "
        
        
        # Create the FourierFeatureEncoding for low-dim grid feats to map
        # high-dim SIREN feats
        
        ffn_list = []
        ffn_sigma_list = []
        self.ffn_sigma_list = []
        for i in range(grid_level):
            ffn_A = torch.randn((feature_dims,ffenc_dims[2+i]),requires_grad=True) # * base_sigma * exp_sigma ** i
            ffn_list.append(ffn_A)
            self.ffn_sigma_list.append(base_sigma * exp_sigma ** i)
        self.register_buffer("ffn_A", torch.stack(ffn_list,dim=0))
        ### The low-frequency MLP part is handled with fourier feature encoding
        for layer in range(0,self.num_sin_layers - 1):
            out_dim = ffenc_dims[layer+1]
            setattr(self, "ff_lin" + str(layer), nn.Linear(ffenc_dims[layer],out_dim))
        
        self.sin_w0 = self.multires-1
        self.sin_w0_high = 2*(self.multires-1)
        self.sin_activation_high = Sine(w0=self.sin_w0_high)
        self.init_ffenc()
        ### Some SIREN initialization stuff 

    def init_ffenc(self):
        for layer in range(0, self.num_sin_layers-1):
            lin = getattr(self, "ff_lin" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0)
    def forward(self, in_pos):
        """
            in_pos: [N, 3], in [-bound, bound] - Index position in VoxelGrid
            in_pos (for grid features) should always be located in [0.0, 1.0]
            x (for Fourier Features) should always be located in [-1.0, 1.0] as they are sine,cosine embeddings of the input
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
            fourier_lin= getattr(self, "ff_lin" + str(layer))
            x = fourier_lin(x)
            # Apply embedding function to input instead of SIREN activation
            x = self.ff_enc.embed(x)

            if layer > 0:
                x = grid_x[layer-1] + x
                fourier_lin_high = getattr(self, "ff_lin_high" + str(layer-1))
                x_high = fourier_lin_high(x)
                x_high = self.ff_enc.embed(x_high)
                x_out = x_out + x_high

        x = x_out

        return x