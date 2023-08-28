
"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    (using appropriate Sine Functions - SIREN branch)
    Then we make multiple levels of encodings and get a multi-resolution embeddings that decompose the space-frequency atributes of the
    coordinates input signal to IDR 
    This is inspired from UBC-Vision NFFB FFB encoder  Look at :https://arxiv.org/abs/2212.01735
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Models Import
from model.embeddings.frequency_enc import FourierFeature as FFenc
from model.embeddings.frequency_enc import PositionalEncoding
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
from model.embeddings.tcnn_src.Sine import *
from model.embeddings.style_tranfer.styleMod import StyleMod
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FourierFilterBanks(nn.Module):

    def __init__(self, GridEncoderNetConfig,freq_enc_type,has_out,bound,layers_type):
        super(FourierFilterBanks, self).__init__()
        
        self.bound = bound
        self.include_input = GridEncoderNetConfig['include_input']
        self.num_inputs = GridEncoderNetConfig['in_dim']
        self.n_levels = GridEncoderNetConfig['n_levels']
        self.max_points_per_level = GridEncoderNetConfig['max_points_per_level']
        
        
        
        # Initi Encoders #
        
        "Multi-Res HashGrid -> Spatial Coord Encoding"
        self.grid_levels = int(self.n_levels)
        print(f"Grid encoder levels: {self.grid_levels}")
        
        self.grid_enc =  MultiResHashGridMLP(self.include_input, self.num_inputs,
                                                self.n_levels, self.max_points_per_level,
                                                GridEncoderNetConfig['log2_hashmap_size'],
                                                GridEncoderNetConfig['base_resolution'],
                                                GridEncoderNetConfig['desired_resolution'])
        
        
        "Fourier Features Network -> Frequency Encoding"
        ff_enc_list = []
        if freq_enc_type == 'FourierFeatureNET':
            for i in range(0,self.grid_levels):
                ffenc_layer = FFenc(channels=self.max_points_per_level, 
                                    sigma = GridEncoderNetConfig['base_sigma']*GridEncoderNetConfig['exp_sigma']**i,
                                    input_dims=nffb_lin_dims[i+1],include_input=True)
                ff_enc_list.append(ffenc_layer)
        elif freq_enc_type == 'PositionalEncodingNet':
            for i in range(0,self.grid_levels):
                posenc_layer = PositionalEncoding(include_input=self.include_input,
                                                input_dims=self.max_points_per_level,
                                                max_freq_log2=GridEncoderNetConfig['log2_hashmap_size'],
                                                num_freqs=self.n_levels,
                                                log_sampling=True,
                                                periodic_fns=[torch.sin, torch.cos])
                ff_enc_list.append(posenc_layer)

        
        
        print(f"FFB Encoder Fourier Feature Filters: {self.grid_levels}")
        nffb_lin_dims = [self.num_inputs] + [ff_enc_list[-1].embeddings_dim]*self.grid_levels
        self.nffb_lin_dims = nffb_lin_dims
        self.ff_enc = nn.ModuleList(ff_enc_list)

        """ The Low - Frequency MLP part """
        
        self.n_nffb_layers = len(nffb_lin_dims)
        assert self.n_nffb_layers >= 5, "The NFFB  should have more than 5 layers"
        # Input layer 
        setattr(self, "ff_lin" + str(0), nn.Linear(nffb_lin_dims[0], nffb_lin_dims[1]))
        for layer in range(1, self.n_nffb_layers - 1):
            setattr(self, "ff_lin" + str(layer), nn.Linear(nffb_lin_dims[layer], nffb_lin_dims[layer + 1]))
        
        """ Initialize parameters for Output Linear Layers - SIREN or ReLU (if has_out)"""
        # SDF network meaning we don't need to change the sine frequency(omega) for each layer -> ReLU is able to approximate the SDF but Wavelet need sine activation
        if layers_type == 'SIREN':
            self.sin_w0 = np.pi * (self.n_levels*self.max_points_per_level)
            self.sin_w0_high = 2*self.sin_w0
            self.sin_activation = Sine(w0=self.sin_w0)
            self.sin_activation_high = Sine(w0=self.sin_w0_high)
            self.lin_activation = self.sin_activation
            self.init_SIREN()
        elif layers_type  == 'ReLU':    
            self.init_ReLU()
            self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=False)
            self.lin_activation = self.relu
        out_layer_width = self.nffb_lin_dims[-1]
        """ The ouput layers if SIREN branch selected or not - High Frequencies are Computed using Siren Layers Coherently with Fourier Grid Features """
        self.has_out = has_out
        if has_out:
            if self.include_input:

                self.embeddings_dim = self.nffb_lin_dims[-1]  + self.num_inputs
                """ The HIGH - Frequency MLP part """
                for layer in range(0, self.grid_levels):
                    setattr(self, "out_lin" + str(layer), nn.Linear(out_layer_width, self.nffb_lin_dims[-1]))
                
                if layers_type  == 'SIREN':  
               
                    self.out_layer = nn.Linear(out_layer_width,self.nffb_lin_dims[-1]).to(device)
                    self.out_activation = Sine(w0=self.sin_w0_high)
                    self.init_SIREN_out()
                elif layers_type  == 'ReLU':
                    self.out_layer = nn.Linear(out_layer_width,self.nffb_lin_dims[-1]).to(device)
                    self.out_activation = nn.LeakyReLU(negative_slope=0.01,inplace=False)
                    self.init_ReLU_out()
            else:
                self.out_layer = nn.Linear(out_layer_width,self.nffb_lin_dims[-1]).to(device)
        else:
            if self.include_input:
                self.embeddings_dim = self.nffb_lin_dims[-1] + self.num_inputs
            else:
                self.embeddings_dim = self.nffb_lin_dims[-1] 
        
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
            Inputs:
                x: [N, 3] - Input points 3D in [-scale,scale]
            Ouputs:
                out: (N,3+feature_Vector_size), embeddings
        """
        x = input / self.bound  # Bound the input between [-1,1]
        input = (input + self.bound) / (2 * self.bound)
        """ Feature Modulation Network Filter Banks Style Transfer """
        self.styleTransferBlock = StyleMod(x.shape[0],self.embeddings_dim-self.num_inputs)
        # Compute HashGrid and split it to it's multiresolution levels
        augmented_grid_x = self.grid_enc(input)
        grid_x = augmented_grid_x[..., x.shape[-1]:]
        grid_x = grid_x.view(-1, self.grid_levels, self.max_points_per_level)
        grid_x = grid_x.permute(1, 0, 2)
        # Embeddings_list corresponds to the indermediated outputs O1,O2,O3... in the paper #
        embeddings_list = []
        for i in range(self.grid_levels):
            grid_ff_output = self.ff_enc[i](grid_x[i])
            embeddings_list.append(grid_ff_output)
        embeddings_list = torch.stack(embeddings_list,dim=0).to(device=input.device)
        if self.has_out:
            if self.include_input:
                x_out = torch.zeros(x.shape[0],self.embeddings_dim-self.num_inputs,device=input.device)
            else:
                x_out = torch.zeros(x.shape[0],self.embeddings_dim-self.num_inputs,device=input.device)
        else:
            features_list = []
        """ Grid Fourier Encoding """
        for layer in range(0,self.n_nffb_layers-1):
            ff_lin = getattr(self,'ff_lin' + str(layer)).to(input.device)
            x = ff_lin(x)
            x = self.lin_activation(x)
            
            if layer > 0:
                k = int(self.nffb_lin_dims[-1])
                embed_Feat = embeddings_list[layer-1] + x
                # Style Modulation #
                #embed_Feat = self.styleTransferBlock(x,embed_Feat)
                if self.has_out:
                    # For Extended High Frequency MLP Layers # 
                    out_layer = getattr(self,"out_lin" + str(layer-1)).to(input.device)
                    
                    x_high = out_layer(embed_Feat)
                    x_high = self.out_activation(x_high)
                    

                    x_out = x_out + x_high
                else:
                    features_list.append(embed_Feat)
       

        if self.has_out:
            x_out = x_out/self.grid_levels
            x = torch.cat([input,x_out],dim=-1)
        else:
            feats = torch.zeros(x.shape[0],self.embeddings_dim-self.num_inputs,device=input.device)
            for i in range(len(features_list)):
                feats += features_list[i]
            x = torch.cat([input,feats/self.grid_levels],dim=-1)
        out_feat = x
        return out_feat
    
    """Functions Used for RELU layers"""
    def init_ReLU(self):
        for layer in range(0, self.n_nffb_layers - 1):
            lin = getattr(self, "ff_lin" + str(layer))
            nn.init.kaiming_normal_(lin.weight, nonlinearity='leaky_relu')
            nn.init.constant_(lin.bias, 0)
            

    def init_ReLU_out(self):
        for layer in range(0, self.n_nffb_layers - 1):
            lin = getattr(self, "out_lin" + str(layer))
            nn.init.kaiming_normal_(lin.weight, nonlinearity='leaky_relu')
            nn.init.constant_(lin.bias, 0)
            

    """Functions Used for SIREN Layers"""
    def init_SIREN(self):
        for layer in range(0, self.n_nffb_layers-1):
            lin = getattr(self, "ff_lin" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0)
    def init_SIREN_out(self):
        for layer in range(0, self.n_nffb_layers-1):
            lin = getattr(self, "out_lin" + str(layer)) 
            sine_init(lin,self.sin_w0_high)

    
        
    # optimizer utils
    def get_optimizer(self,lr,weight_decay):
        return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)  