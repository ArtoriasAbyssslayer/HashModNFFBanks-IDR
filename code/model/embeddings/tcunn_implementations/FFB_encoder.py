import torch
import torch.nn as nn
import tinycudann as tcnn 
import math 
from .Sine import *
from model.embeddings.frequency_enc import FourierFeature as FFenc
from .hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as HashEncoderTcnn


"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    Then we make multiple level of encodings and get a multi-resolution embeddings and sum them up to

"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FFB_encoder(nn.Module):
    def __init__(self,HashGridEncoderConfig,bound):
        super().__init__()
        self.in_dim = HashGridEncoderConfig['in_dim']
        
        self.bound = bound 
        self.n_levels = HashGridEncoderConfig['n_levels']
        multires = self.n_levels
        self.feature_dims = HashGridEncoderConfig['max_points_per_level']
        # FourierFeatureEncoding init
        n_frequencies = self.n_levels
        include_input = HashGridEncoderConfig['include_input']
        FFenc_kwargs = {
            'channels': HashGridEncoderConfig['network_dims'][0],
            'sigma': 1.0,
            'input_dims': self.in_dim,
            'include_input': include_input
        }
        self.ff_enc = FFenc(**FFenc_kwargs)
        ffenc_dims = [self.in_dim]+[128]*n_frequencies 
        self.num_sin_layers = len(ffenc_dims)
        assert self.num_sin_layers > 3, "The layer number (SIREN branch) should be greater than 3 "
        # HashTcnn Encoder init
        grid_level = int(self.num_sin_layers - 1)        
        self.grid_encoder = HashEncoderTcnn(**HashGridEncoderConfig)
        self.grid_level = grid_level
        print(f"Grid encoder levels: {self.grid_level}")

        # Create the FourierFeatureEncoding for low-dim grid feats to map
        # high-dim Positional Fourier feats - or  SIREN features
        
        ffn_list = []
        ffn_sigma_list = []
        self.ffn_sigma_list = []
        for i in range(grid_level):
            ffn_A = torch.randn((self.feature_dims,ffenc_dims[i+1]),requires_grad=True) # * base_sigma * exp_sigma ** i
            ffn_list.append(ffn_A)
            self.ffn_sigma_list.append(HashGridEncoderConfig['base_sigma'] * HashGridEncoderConfig['exp_sigma'] ** i)
        self.register_buffer("ffn_A", torch.stack(ffn_list,dim=0))
        ## The low-frequency MLP part is handled with fourier feature encoding
        for layer in range(0,self.num_sin_layers - 1):
            out_dim = ffenc_dims[layer+1]
            setattr(self, "ff_lin" + str(layer), nn.Linear(ffenc_dims[layer],out_dim).to(device=DEVICE))
        self.sin_w0 = n_frequencies
        self.init_ffenc()  
        # The high frequency MLP part
        for layer in range(0, grid_level):
            setattr(self,"ff_lin_high" + str(layer), nn.Linear(ffenc_dims[layer+1],ffenc_dims[layer+1]*2+2*self.in_dim ).to(device=DEVICE))
        self.ffenc_dims = ffenc_dims
        self.sin_w0_high = 2*n_frequencies
        self.sin_activation = Sine(w0=self.sin_w0)
        self.sin_activation_high = Sine(w0=self.sin_w0_high)
        self.init_ffenc_high()
        if include_input:
            self.embeddings_dim = 2*ffenc_dims[-1] + self.in_dim
        else:
            self.embeddings_dim = ffenc_dims[-1]
        ### Some SIREN initialization stuff 
        self.include_input = include_input

    def init_ffenc(self):
        for layer in range(0, self.num_sin_layers-1):
            lin = getattr(self, "ff_lin" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0)
    def init_ffenc_high(self):
        for layer in range(0, self.num_sin_layers-1):
            lin = getattr(self, "ff_lin_high" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0_high)
        
    def forward(self, in_pos):
        """
            in_pos: [N, 3], in [-bound, bound] - Index position in VoxelGrid
            in_pos (for grid features) should always be located in [0.0, 1.0]
            x (for Fourier Features) should always be located in [-1.0, 1.0] as they are sine,cosine embeddings of the input
        """
        
        x = in_pos / self.bound								# to [-1, 1]
        
        in_pos = (in_pos + self.bound) / (2 * self.bound) 	# to [0, 1]
    
        augmented_grid_x = self.grid_encoder(in_pos)
        grid_x = augmented_grid_x[..., x.shape[-1]:]
        grid_x = grid_x.view(-1, self.grid_level, self.feature_dims)
        grid_x = grid_x.permute(1, 0, 2)

        ffn_A_list = []
        for i in range(self.grid_level):
            ffn_A_list.append(self.ffn_A[i] * self.ffn_sigma_list[i])
        ffn_A = torch.stack(ffn_A_list, dim=0).to(in_pos.device)

        #This would be applied with FourierFeaturesEncoding
        grid_x = torch.bmm(grid_x, 2 * math.pi * ffn_A)
        grid_x = torch.sin(grid_x)

        embed_buff = torch.zeros(x.shape[0], (int)((self.embeddings_dim - self.in_dim)/2), device=in_pos.device)

        ### Grid encoding
        # x = self.ff_enc.embed(x)
        # x = x[..., x.shape[-1]-3:x.shape[-1]]
        for layer in range(0, self.num_sin_layers - 1):
                ff_lin = getattr(self, "ff_lin" + str(layer))
                x = ff_lin(x)
                x = self.sin_activation(x)
                x_low = torch.zeros_like(embed_buff)
                x_high = torch.zeros_like(embed_buff)
                if layer > 0:
                    x = grid_x[layer-1] + x

                    sin_lin_high = getattr(self, "ff_lin_high" + str(layer-1))
                    x_high = sin_lin_high(x)
                    x_high = self.sin_activation_high(x_high)
                    x_high = torch.split(x_high, self.ffenc_dims[layer+1], dim=-1)
                    x_low = x_high[0] 
                    x_high = x_high[1] 
                    embed_buff = embed_buff+  x_low + x_high
                
        x_out = torch.cat([x,embed_buff],dim=1)        
        if self.include_input:
            x_out = torch.cat([in_pos,x_out], dim=-1)
        torch.cuda.empty_cache()
        return x_out