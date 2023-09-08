import torch
import torch.nn as nn
import tinycudann as tcnn 
import math 
from model.embeddings.Sine import *
from .hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as HashEncoderTcnn


"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Based on UBC-Vision NFFB https://arxiv.org/abs/2212.01735

"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FFB_encoder(nn.Module):
    def __init__(self,HashGridEncoderConfig,bound,has_out=True):
        super().__init__()
        self.in_dim = HashGridEncoderConfig['in_dim']
        self.bound = bound 
        self.n_levels = HashGridEncoderConfig['n_levels']
        self.feature_dims = HashGridEncoderConfig['max_points_per_level']
        self.has_out = has_out
        # FourierFeatureEncoding init
        n_frequencies = self.n_levels
        include_input = HashGridEncoderConfig['include_input']
        # NFFB layers width selected to be 128 as TCNN layers support caps to that number of neurons(and no uneeded decodings are required)
        ffenc_dims = [self.in_dim]+[128]*n_frequencies 
        self.ffenc_dims = ffenc_dims
        self.n_nffb_layers = len(ffenc_dims)
        assert self.n_nffb_layers > 3, "The layer number (SIREN branch) should be greater than 3 "
        
        """Multiresolution HashGrid TCNN Variant Init"""
        grid_level = int(self.n_nffb_layers - 2)        
        self.grid_encoder = HashEncoderTcnn(**HashGridEncoderConfig)
        self.grid_level = grid_level
        print(f"Grid encoder levels: {self.grid_level}")
        "Fourier Features NTK init"
        ffn_list = []
        self.ffn_sigma_list = []
        for i in range(grid_level):
            ffn_A = torch.randn((self.feature_dims,ffenc_dims[i+2]),requires_grad=True)*HashGridEncoderConfig['base_sigma'] * HashGridEncoderConfig['exp_sigma'] ** i
            ffn_list.append(ffn_A)
        self.register_buffer("ffn_A", torch.stack(ffn_list,dim=0))
        
        "Low Frequency MLP layers"
        for layer in range(0,self.n_nffb_layers - 1):
            out_dim = ffenc_dims[layer+1]
            setattr(self, "ff_lin" + str(layer), nn.Linear(ffenc_dims[layer],out_dim).to(device=DEVICE))
        self.sin_w0 = n_frequencies**self.feature_dims - n_frequencies # 36 - 6 = 30 based on SIREN paper
        self.sin_activation = Sine(w0=self.sin_w0)
        self.init_ffenc()  
        "High Frequency MLP layers"
        if has_out == True:
            for layer in range(0, grid_level):
                self.out_dim = ffenc_dims[-1] # If it seems slow I should add size factor to match the IDR feature vector width [128]*2
                setattr(self,"ff_lin_high" + str(layer), nn.Linear(ffenc_dims[layer+1],self.out_dim).to(device=DEVICE))
            
            # For SDF embedding network High Frequency sine activation is the same with Low Frequency sine activation
            self.sin_w0_high = self.sin_w0 
            self.sin_activation_high = Sine(w0=self.sin_w0_high)
            self.init_ffenc_high()
        if include_input:
            self.embeddings_dim = ffenc_dims[-1] + self.in_dim
        else:
            self.embeddings_dim = ffenc_dims[-1]
        ### Some SIREN initialization stuff 
        self.include_input = include_input

    def init_ffenc(self):
        for layer in range(0, self.n_nffb_layers-1):
            lin = getattr(self, "ff_lin" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0)
    def init_ffenc_high(self):
        for layer in range(0, self.n_nffb_layers-1):
            lin = getattr(self, "ff_lin_high" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0_high)
        
    def forward(self, in_pos,compute_grad=False):
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

        if self.has_out:
            embed_buff = torch.zeros(x.shape[0], self.embeddings_dim - self.in_dim, device=in_pos.device)
        else :
            embed_buff = []
        ### Wavelet-Like Encoding
        for layer in range(0, self.n_nffb_layers - 1):
                ff_lin = getattr(self, "ff_lin" + str(layer))
                x = ff_lin(x)
                x = self.sin_activation(x)
                if layer > 0:
                    x = grid_x[layer-1] + x
                    if self.has_out:
                        
                        sin_lin_high = getattr(self, "ff_lin_high" + str(layer-1))
                        x_high = sin_lin_high(x)
                        x_high = self.sin_activation_high(x_high)
                        embed_buff = embed_buff +  x_high
                    else:
                        embed_buff.append(x)
                        
        if self.include_input:
            x_out = torch.cat([in_pos,embed_buff/self.grid_level], dim=-1)
        else:
            x_out = torch.cat([in_pos])

        return x_out