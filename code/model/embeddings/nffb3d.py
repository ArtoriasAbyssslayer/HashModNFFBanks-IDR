
"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    (using appropriate Sine Functions - SIREN branch)
    Then we make multiple levels of encodings and get a multi-resolution embeddings that decompose the space-frequency atributes of the
    coordinates input signal to IDR 
    
    This is inspired(with many modifications) from UBC-Vision NFFB FFB encoder paper which is explained here :https://arxiv.org/abs/2212.01735
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Models Import
from model.embeddings.frequency_enc import FourierFeature as FFenc
from model.embeddings.frequency_enc import PositionalEncoding
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
from model.embeddings.Sine import *
from model.embeddings.style_Attention.styleMod import StyleAttention,StyleModulation
# from model.embeddings.style_Attention.multihead_attention import MultiHeadAttentionModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FourierFilterBanks(nn.Module):

    def __init__(self, GridEncoderNetConfig,freq_enc_type,has_out,bound,layers_type,style_modulation=False):
        super(FourierFilterBanks, self).__init__()
        self.bound = bound
        self.skip_in = [4]
        self.include_input = GridEncoderNetConfig['include_input']
        self.num_inputs = GridEncoderNetConfig['in_dim']
        self.n_levels = GridEncoderNetConfig['n_levels']
        self.max_points_per_level = GridEncoderNetConfig['max_points_per_level']
        self.network_dims = GridEncoderNetConfig['network_dims']
        self.modulationApplied = style_modulation
        # Initi Encoders #
        "Multi-Res HashGrid -> Spatial Coord Encoding"
        self.grid_levels = int(self.n_levels)
        print(f"Grid encoder levels: {self.grid_levels}")
        self.grid_enc =  MultiResHashGridMLP(self.include_input, self.num_inputs,
                                                self.n_levels, self.max_points_per_level,
                                                GridEncoderNetConfig['log2_hashmap_size'],
                                                GridEncoderNetConfig['base_resolution'],
                                                GridEncoderNetConfig['desired_resolution']).to(device)
        
        "Fourier Features Network -> Frequency Encoding"
        ff_enc_list = []
        # Number of decomposed frequency levels are hardcoded in order to match the Feature Vector Size Dimensions
        freq_num = self.max_points_per_level**(self.n_levels+1)
        if freq_enc_type == 'FourierFeatureNET':
            for i in range(self.grid_levels):
                ffenc_layer = FFenc(input_dims=self.max_points_per_level, 
                                    sigma = GridEncoderNetConfig['base_sigma']*GridEncoderNetConfig['exp_sigma']**i,
                                    num_channels=freq_num,
                                    include_input=False)
                ff_enc_list.append(ffenc_layer)
        elif freq_enc_type == 'PositionalEncodingNET':
            for i in range(self.grid_levels):
                posenc_layer = PositionalEncoding(include_input=self.include_input,
                                                input_dims=self.max_points_per_level,
                                                max_freq_log2=self.n_levels-1,
                                                num_freqs= self.n_levels,
                                                log_sampling=True,
                                                periodic_fns=[torch.sin, torch.cos])
                ff_enc_list.append(posenc_layer)
        # IDR feature vector size in network_dims[-1]= 25 and is hardcoded here (fix this)
        if freq_enc_type == 'FourierFeatureNET':
            nffb_lin_dims = [self.num_inputs] + [ff_enc_list[-1].embeddings_dim]*(self.grid_levels-1)
        else:
            nffb_lin_dims = [self.num_inputs] + [ff_enc_list[-1].embeddings_dim]*(self.grid_levels-1)
        self.nffb_lin_dims = nffb_lin_dims
        self.ff_enc = nn.Sequential(*ff_enc_list).to(device)
        """ The Low - Frequency MLP part """
        self.n_nffb_layers = len(nffb_lin_dims)
        print(f"FFB Encoder Fourier Grid Filters: {self.n_nffb_layers}")
        assert self.n_nffb_layers >= 3, "The NFFB  should have more than 5 layers"
        # Input layer 
        setattr(self, "ff_lin" + str(0), nn.Linear(nffb_lin_dims[0], nffb_lin_dims[1]))
        for layer in range(1, self.n_nffb_layers - 1):
            setattr(self, "ff_lin" + str(layer), nn.Linear(nffb_lin_dims[layer], nffb_lin_dims[layer + 1]))
        """ Initialize parameters for Linear Layers"""
        # SDF network meaning we don't need to change the sine frequency(omega) for each layer -> ReLU is able to approximate the SDF but Wavelet need sine activation
        if layers_type == 'SIREN':
            self.sin_w0 = self.n_levels**self.max_points_per_level - self.n_levels # 36-6 = 30 SIREN frequency as described in the paper
            self.sin_w0_high = self.sin_w0 # Keep the same frequency for High Frequency MLP Layers because SDF is a stationary function
            self.sin_activation = Sine(w0=self.sin_w0)
            self.sin_activation_high = Sine(w0=self.sin_w0_high)
            self.lin_activation = self.sin_activation
            self.init_SIREN()
        elif layers_type  == 'ReLU':    
            self.init_ReLU()
            self.lin_activation = nn.LeakyReLU(negative_slope=1e-2,inplace=False)
        out_layer_width = self.nffb_lin_dims[-1]
        self.feature_Vector_size  = out_layer_width
        # The ouput layers if SIREN branch selected or not - High Frequencies are Computed using Siren Layers Coherently with Fourier Grid Features 
        self.has_out = has_out
        
        if self.include_input:
            self.embeddings_dim = self.nffb_lin_dims[-1] + self.num_inputs
        else:
            self.embeddings_dim = self.nffb_lin_dims[-1] 
        if has_out:
        
        
            """ The HIGH - Frequency MLP part """
            for layer in range(0, self.grid_levels):
                setattr(self, "out_lin" + str(layer), nn.Linear(out_layer_width, self.nffb_lin_dims[-1]))
            
            if layers_type  == 'SIREN':  
            
                self.out_layer = nn.Linear(out_layer_width,self.nffb_lin_dims[-1])
                self.out_activation = Sine(w0=self.sin_w0_high)
                self.init_SIREN_out()
            elif layers_type  == 'ReLU':
                self.out_layer = nn.Linear(out_layer_width,self.nffb_lin_dims[-1])
                self.out_activation = nn.LeakyReLU(negative_slope=1e-2,inplace=False)
                self.init_ReLU_out()
        else:
            self.out_layer = nn.Linear(out_layer_width,self.nffb_lin_dims[-1])
        
        self.StyleAttentionBlock = StyleAttention(self.num_inputs,self.feature_Vector_size)
        self.StyleModulationBlock = StyleModulation(self.n_levels,self.feature_Vector_size)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input: torch.Tensor,compute_grad=False) -> torch.Tensor:
        """
            Inputs:
                x: [N, 3] - Input points 3D in [-scale,scale]
            Ouputs:
                out: (N,3+feature_Vector_size), embeddings
        """

        
        x = input / self.bound  # Bound the input between [-1,1]
        input = (input + self.bound) / (2 * self.bound) 


        # Compute HashGrid and split it to it's multiresolution levels
        augmented_grid_x = self.grid_enc(input)
        grid_x = augmented_grid_x[..., x.shape[-1]:]
        grid_x = grid_x.view(-1, self.grid_levels, self.max_points_per_level)
        grid_x = grid_x.permute(1, 0, 2).to(input.device)
        # Embeddings_list corresponds to the indermediated outputs O1,O2,O3... in the paper #
        embeddings_list = []
        for i in range(self.grid_levels):
            grid_ff_output = self.ff_enc[i](grid_x[i])
            embeddings_list.append(grid_ff_output)
        embeddings_list = torch.stack(embeddings_list,dim=0).to(device=input.device)


        if self.has_out:
            x_out = torch.zeros(x.shape[0],self.embeddings_dim-self.num_inputs,device=input.device)
        else:
            features_list = []
        """                         Style Modulation
                    Assuming style vector is the embed Feat - which is Fourier Feature Grid
                    and the feature vector is the x which is derived from the hash grid
                    Essentially map better ntk fourier features to the hash grid features
        """
        # self.StyleModulationBlock(x,embeddings_list)
        """ Grid Fourier Encoding """
        
        for layer in range(0,self.n_nffb_layers-1):
            ff_lin = getattr(self,'ff_lin' + str(layer)).to(x.device)
            x = ff_lin(x)
            x = self.lin_activation(x)     
            if layer > 0:
                " Style Attention " 
                if self.modulationApplied:
                    demodulated_embed_Feat = self.StyleAttentionBlock(input,embeddings_list[layer-1])
                    embed_Feat = demodulated_embed_Feat  + x
                else:
                    embed_Feat = embeddings_list[layer-1] + x
              
                if self.has_out:
                    # For Extended High Frequency MLP Layers # 
                    out_layer = getattr(self,"out_lin" + str(layer-1)).to(embed_Feat.device)
                    x_high = out_layer(embed_Feat)
                    x_high = self.out_activation(x_high)
                    x_out = x_out + x_high
                else:
                    lin_out = self.out_layer.to(x.device)
                    embed_Feat = lin_out(embed_Feat)
                    features_list.append(embed_Feat)

        if self.has_out:
            x_out = x_out/(self.grid_levels)
            x = torch.cat([input,x_out],dim=-1)
        else:
            feats = torch.zeros(x.shape[0],self.embeddings_dim-self.num_inputs,device=input.device)
            for i in range(len(features_list)):
                feats += features_list[i]
            x = torch.cat([input,feats/(self.grid_levels)],dim=-1)
        return x
    " Functions Used for RELU layers - IGR ReLU -> Results to Smooth SDFs"
    def init_ReLU(self):
        for layer in range(0, self.n_nffb_layers - 1):
            lin = getattr(self, "ff_lin" + str(layer))
            if layer == self.n_nffb_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(self.nffb_lin_dims[layer]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -0.6)
            elif self.n_nffb_layers > 0 and layer == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(self.nffb_lin_dims[-1]))
            elif self.n_nffb_layers > 0 and layer in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.nffb_lin_dims[-1]))
                torch.nn.init.constant_(lin.weight[:, -(self.nffb_lin_dims[0] - 3):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.nffb_lin_dims[-1]))
    print("IGR completed")
    def init_ReLU_out(self):
        for layer in range(self.grid_levels):
            lin = getattr(self, "out_lin" + str(layer))
            if layer == self.n_nffb_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(self.nffb_lin_dims[layer]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -0.6)
            elif self.grid_levels > 0 and layer == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(self.nffb_lin_dims[-1]))
            elif self.grid_levels > 0 and layer in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.nffb_lin_dims[-1]))
                torch.nn.init.constant_(lin.weight[:, -(self.nffb_lin_dims[0] - 3):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.nffb_lin_dims[-1]))
    print("IGR Out Head completed")
    " Functions Used for SIREN Layers -> Results to Sharp SDFs"
    def init_SIREN(self):
        for layer in range(0, self.n_nffb_layers-1):
            lin = getattr(self, "ff_lin" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0)
    def init_SIREN_out(self):
        for layer in range(self.grid_levels):
            lin = getattr(self, "out_lin" + str(layer)) 
            sine_init(lin,self.sin_w0_high)