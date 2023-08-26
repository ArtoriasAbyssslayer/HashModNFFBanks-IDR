
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
import numpy as np
# Models Import
from model.embeddings.frequency_enc import FourierFeature as FFenc
from model.embeddings.frequency_enc import PositionalEncoding
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
from model.embeddings.tcnn_src.Sine import *
from model.embeddings.style_tranfer.styleMod import StyleMod
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FourierFilterBanks(nn.Module):

    def __init__(self, GridEncoderNetConfig,has_out,bound):
        super(FourierFilterBanks, self).__init__()
        
        self.bound = bound
        self.include_input = GridEncoderNetConfig['include_input']
        self.num_inputs = GridEncoderNetConfig['in_dim']
        self.n_levels = GridEncoderNetConfig['n_levels']
        self.max_points_per_level = GridEncoderNetConfig['max_points_per_level']
        self.sin_w0 = np.pi * (self.n_levels**self.max_points_per_level)
        """ Initialize Encoders """ 
        # Multi-Res HashGrid -> Spatial Coord Encoding
        self.grid_levels = int(self.n_levels)
        print(f"Grid encoder levels: {self.grid_levels}")
        
        self.grid_enc =  MultiResHashGridMLP(self.include_input, self.num_inputs,
                                                self.n_levels, self.max_points_per_level,
                                                GridEncoderNetConfig['log2_hashmap_size'],
                                                GridEncoderNetConfig['base_resolution'],
                                                GridEncoderNetConfig['desired_resolution'])

        # Fourier Features NTK Stationary - Frequency Coord Encoding 
        # (Select Half the neurons of the feature vector size -> 1/2 neurons of IDR network for each layer for faster training)
        
        
        #for i in range(0,self.grid_levels):
        #    ffenc_layer = FFenc(channels=self.max_points_per_level, 
        #                        sigma = GridEncoderNetConfig['base_sigma']*GridEncoderNetConfig['exp_sigma']**i,
        #                        input_dims=nffb_lin_dims[i+1],include_input=True)
        #    ff_enc_list.append(ffenc_layer)
        """ Use Positional Encoding for Frequency Encoding ffenc"""
        ff_enc_list = []
        for i in range(0,self.grid_levels):
            posenc_layer = PositionalEncoding(include_input=self.include_input,
                                              input_dims=self.max_points_per_level,
                                              max_freq_log2=self.n_levels-1,
                                              num_freqs=self.n_levels,
                                              log_sampling=True,
                                              periodic_fns=[torch.sin, torch.cos])
            ff_enc_list.append(posenc_layer)
        
        nffb_lin_dims = [self.num_inputs] + [posenc_layer.embeddings_dim]*self.grid_levels
        self.nffb_lin_dims = nffb_lin_dims
        self.ff_enc = nn.ModuleList(ff_enc_list)

        """ The Low - Frequency MLP part """
        
        self.n_nffb_layers = len(nffb_lin_dims)
        assert self.n_nffb_layers >= 5, "The NFFB  should have more than 5 layers"
        # Input layer 
        setattr(self, "ff_lin" + str(0), nn.Linear(nffb_lin_dims[0], nffb_lin_dims[1]))
        for layer in range(1, self.n_nffb_layers - 1):
            setattr(self, "ff_lin" + str(layer), nn.Linear(nffb_lin_dims[layer], nffb_lin_dims[layer + 1]))
        
        """ Initialize parameter if  SIREN Branch is used <-> has_out(bool)"""
        # SDF network meaning we don't need to change the sine frequency(omega) for each layer -> ReLU is able to approximate the SDF but Wavelet need sine activation
        #self.sin_w0_high = self.sin_w0
        #self.sin_activation = Sine(w0=self.sin_w0)
        #self.sin_activation_high = Sine(w0=self.sin_w0_high)
        #self.init_SIREN()

        out_layer_width = self.nffb_lin_dims[-1]
        """ The ouput layers if SIREN branch selected or not - High Frequencies are Computed using Siren Layers Coherently with Fourier Grid Features """
        self.has_out = has_out
        if has_out:
            if self.include_input:

                self.embeddings_dim = self.nffb_lin_dims[-1]  + self.num_inputs
                """ The HIGH - Frequency MLP part """
                for layer in range(0, self.grid_levels):
                    setattr(self, "out_lin" + str(layer), nn.Linear(out_layer_width, self.nffb_lin_dims[-1]))
                #self.init_SIREN_out()
                #self.out_activation = Sine(w0=self.sin_w0_high)
                self.out_layer = nn.Linear(out_layer_width,self.nffb_lin_dims[-1]).to(device)
            else:
                self.out_layer = nn.Linear(out_layer_width,self.nffb_lin_dims[-1]).to(device)
        else:
            if self.include_input:
                self.embeddings_dim = self.nffb_lin_dims[-1] + self.num_inputs
            else:
                self.embeddings_dim = self.nffb_lin_dims[-1] 
        """ Feature Modulation Network Filter Banks Style Transfer """
        self.styleTransferBlock = StyleMod(feature_vector_size =self.embeddings_dim)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
        
            #x = self.sin_activation(x)
            x = torch.nn.functional.relu(x)
            
            if layer > 0:
                k = int(self.nffb_lin_dims[-1])
                embed_Feat = embeddings_list[layer-1] + x
                # Style Modulation #
                #self.styleTransferBlock(x,embed_Feat)
                if self.has_out:
                    # for SIREN BRANCH
                    out_layer = getattr(self,"out_lin" + str(layer-1)).to(input.device)
                    
                    x_high = out_layer(embed_Feat)
                    #x_high = self.out_activation(x_high)
                    x_high = torch.nn.functional.relu(x_high)

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