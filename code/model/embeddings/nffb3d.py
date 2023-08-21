
"""
    FourierFilterBanks is based on the paper Fourier Filter Banks
    Essentially manage the gridEncoding as a low frequency encoding (LPF) and the FourierFeaturesMLP as a high frequency encoding (HPF)
    Then we make multiple level of encodings and get a multi-resolution embeddings and sum them up to

    This is inspired from UBC-Vision NFFB FFB encoder  https://github.com/ubc-vision/NFFB

"""

import torch
import torch.nn as nn

from model.embeddings.frequency_enc import FourierFeature as FFenc
from model.embeddings.hash_encoder.hashgridencoder import MultiResolutionHashEncoderCUDA as MultiResHashGridCUDA
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
from model.embeddings.tcnn.Sine import *
from model.embeddings.stylemod import styleMod
class FourierFilterBanks(nn.Module):

    def __init__(self, GridEncoderNetConfig,has_out,bound):
        super(FourierFilterBanks, self).__init__()
        self.bound = bound
        self.include_input = GridEncoderNetConfig['include_input']
        self.num_inputs = GridEncoderNetConfig['num_inputs']
        self.n_levels = GridEncoderNetConfig['n_levels']
        self.max_points_per_level = GridEncoderNetConfig['max_points_per_level']
        self.sin_w0 = n_frequencies
        idr_dims = GridEncoderNetConfig['network_dims']
        
        """ Initialize Encoders """ 
        # Multi-Res HashGrid -> Spatial Coord Encoding
        self.grid_levels = int(self.n_levels - 1)
        print(f"Grid encoder levels: {self.grid_levels}")
        
        self.grid_enc =  MultiResHashGridMLP(self.include_input, self.num_inputs,
                                                self.n_levels, self.max_points_per_level,
                                                GridEncoderNetConfig['log2_hashmap_size'],
                                                GridEncoderNetConfig['base_resolution'],
                                                GridEncoderNetConfig['desired_resolution'])

        # Fourier Features NTK Stationary - Frequency Coord Encoding 
        # (Select Half the neurons of the feature vector size -> 1/4 neurons of IDR network for each layer for faster training)
        ffenc_dims = [self.num_inputs]+ int(idr_dims[2]/4)*self.grid_levels
        self.ffenc_dims = ffenc_dims
        ff_enc_list = []
        for i in range(self.grid_levels):
            ffenc_layer = FFenc(channels=self.max_points_per_level*ffenc_dims[2+i], 
                                sigma = GridEncoderNetConfig['base_sigma']*GridEncoderNetConfig['exp_sigma']**i,
                                input_dims=ffenc_dims[i+1],include_input=True)
            ff_enc_list.append(ffenc_layer)
        self.ff_enc = nn.ModuleList(ff_enc_list)


        """ The Low - Frequency MLP part """
        
        self.n_ffenc_layers = len(idr_dims)
        assert self.n_ffenc_layers > 6, "The Implicit Network Branch should have more than 6 layers"
        for layer in range(0, self.n_ffenc_layers - 1):
            setattr(self, "ff_lin" + str(layer), nn.Linear(ffenc_dims[layer], ffenc_dims[layer + 1]))
        
        """ Initialize parameter if  SIREN Branch is used <-> has_out(bool)"""
        self.sin_w0_high = 2*n_frequencies
        self.sin_activation = Sine(w0=self.sin_w0)
        self.sin_activation_high = Sine(w0=self.sin_w0_high)
        self.init_SIREN_high()


        """ The ouput layers if SIREN branch selected or not - High Frequencies are Computed using Siren Layers Coherently with Fourier Grid Features """
        if has_out:
            if self.include_input:

                self.embeddings_dim = self.ffenc_dims[-1] + self.num_inputs
                
                ### SIREN BRANCH ### 
                for layer in range(0, grid_level):
                    setattr(self, "out_lin" + str(layer), nn.Linear(ffenc_dims[layer + 1], self.embeddings_dim))

                    
                self.sin_w0_high = network_config["w1"]
                self.init_siren_out()
                self.out_activation = Sine(w0=self.sin_w0_high)
                self.out_layer = nn.Linear(self.ffenc_dims[-1],self.embeddings_dim)
            else:
                self.out_layer = nn.Linear(self.ffenc_dims[-2],self.ffenc_dims[-1])
        else:
            if self.include_input:
                self.embeddings_dim = ffenc_dims[-1] * grid_level + num_inputs
            else:
                self.embeddings_dim = ffenc_dims[-1] * grid_level
        """ Feature Modulation Network Filter Banks Style Transfer """
        self.styleTransferBlock = styleMod(feature_vector_size = ffenc_dims[-1]*grid_level + num_inputs)
            


    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        IndermediateOutputs = []
        x = input / self.bound  # Bound the input between [-1,1]
        input = (input + self.bound) / (2 * self.bound)
        
        # Compute HashGrid and split it to it's multiresolution levels
        augmented_grid_x = self.grid_enc(input)
        grid_x = augmented_grid_x[..., x.shape[-1]:]
        grid_x = grid_x.view(-1, self.grid_level, self.feature_dims)
        grid_x = grid_x.permute(1, 0, 2)
        
        embeddings_list = []
        for i in range(self.grid_level):
            grid_ff_output = self.ff_enc[i](grid_x[i])
            embeddings_list.append(grid_ff_output)
        
        if self.has_out:
            if self.include_input:
                x_out = torch.zeros(x.shape[0], self.embeddings_dim,device=in_pos.device)
            else:
                x_out = torch.zero(x.shape[0],self.embeddings_dim,device=in_pos.device)
        else:
            features_list = []
        """ Grid Fourier Encoding """
        for layer in range(self.n_ffenc_layers):
            ff_lin = getattr(self,'ff_lin' + str(layer))
            x = ff_lin(x)
            # for SIREN BRANCH
            # x = self.sin_activation(x)
            
            if layer > 0:

                embed_Feat = embeddings_list[layer-1] + x
                # Style Modulation #
                self.styleTransferBlock(x,embed_Feat)
                if self.has_out:
                    out_layer = getattr(self,"out_lin" + str(layer-1))
                    x_high = out_layer(embed_Feat)
                    x_high = self.out_activation(x_high)

                    x_out = x_out + x_high
                else:
                    features_list.append(embed_Feat)
                

        if self.has_out:
            x = x_out
        else:
            x = features_list
        # TODO - Remove Non Redacted Code        
        # # Sum Fourier features with L_i MLP outputs
        # for i in range(self.n_levels):
        #     x = self.mlp_layers[i](x)
        #     if i == 0:
        #         IndermediateOutputs.append(self.Fourier_Grid_features[i].embed(x))
        #     else:
        #         x_high = self.Fourier_Grid_features[i-1](self.Fourier_Grid_features[i=1].embed(x))
        #         IndermediateOutputs.append(x_high)

        # x = torch.sum(torch.stack(IndermediateOutputs), dim=0)
        # output = self.out_layer(x)
        # if self.include_input:
        #     output_dim = output.shape[-1] + input.shape[-1]
        #     self.register_parameter("output_dim", torch.nn.Parameter(torch.tensor(output_dim)))

        return output

    """Functions Used for SIREN Layers"""
    def init_SIREN(self):
        for layer in range(0, self.num_sin_layers-1):
            lin = getattr(self, "ff_lin" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0)
    def init_SIREN_high(self):
        for layer in range(0, self.num_sin_layers-1):
            lin = getattr(self, "ff_lin_high" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0_high)

    def init_SIREN_out(self):
        for layer in range(0, self.grid_level):
            lin = getattr(self, "out_lin" + str(layer))

            sine_init(lin, w0=self.sin_w0_high)