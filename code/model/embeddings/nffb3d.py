
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
# Models Import
from model.embeddings.frequency_enc import FourierFeature as FFenc
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
        self.sin_w0 = self.n_levels**self.max_points_per_level + self.num_inputs
        idr_dims = GridEncoderNetConfig['network_dims']
        self.idr_dims = idr_dims
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
        ffenc_dims = [self.num_inputs]+ [int((idr_dims[-1]-1)/2)]*self.grid_levels
        self.ffenc_dims = ffenc_dims
        ff_enc_list = []
        for i in range(0,self.grid_levels):
            ffenc_layer = FFenc(channels=self.max_points_per_level, 
                                sigma = GridEncoderNetConfig['base_sigma']*GridEncoderNetConfig['exp_sigma']**i,
                                input_dims=ffenc_dims[i+1],include_input=True)
            ff_enc_list.append(ffenc_layer)
        self.ff_enc = nn.ModuleList(ff_enc_list)


        """ The Low - Frequency MLP part """
        
        self.n_ffenc_layers = self.grid_levels
        assert self.n_ffenc_layers >= 6, "The Implicit Network Branch should have more than 5 layers"
        for layer in range(0, self.n_ffenc_layers - 1):
            setattr(self, "ff_lin" + str(layer), nn.Linear(ffenc_dims[layer], ffenc_dims[layer + 1]))
        
        """ Initialize parameter if  SIREN Branch is used <-> has_out(bool)"""
        self.sin_w0_high = 2*self.sin_w0
        self.sin_activation = Sine(w0=self.sin_w0)
        self.sin_activation_high = Sine(w0=self.sin_w0_high)
        self.init_SIREN()


        """ The ouput layers if SIREN branch selected or not - High Frequencies are Computed using Siren Layers Coherently with Fourier Grid Features """
        self.has_out = has_out
        if has_out:
            if self.include_input:

                self.embeddings_dim = self.ffenc_dims[-1] + self.num_inputs
                
                ### SIREN BRANCH ### 
                for layer in range(0, self.grid_levels):
                    setattr(self, "out_lin" + str(layer), nn.Linear(ffenc_dims[layer + 1], self.embeddings_dim))

                
                self.init_SIREN_out()
                self.out_activation = Sine(w0=self.sin_w0_high)
                self.out_layer = nn.Linear(self.ffenc_dims[-1],self.embeddings_dim).to(device)
            else:
                self.out_layer = nn.Linear(self.ffenc_dims[-2],self.ffenc_dims[-1]).to(device)
        else:
            if self.include_input:
                self.embeddings_dim = self.ffenc_dims[-1] + self.num_inputs
            else:
                self.embeddings_dim = self.ffenc_dims[-1] 
        """ Feature Modulation Network Filter Banks Style Transfer """
        self.styleTransferBlock = StyleMod(feature_vector_size =self.embeddings_dim )
    
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
                x_out = torch.zeros(x.shape[0], self.embeddings_dim,device=input.device)
            else:
                x_out = torch.zeros(x.shape[0],self.embeddings_dim-self.num_inputs,device=input.device)
        else:
            features_list = []
        """ Grid Fourier Encoding """
        for layer in range(0,self.n_ffenc_layers-1):
            ff_lin = getattr(self,'ff_lin' + str(layer)).to(input.device)
            x = ff_lin(x)
            #x = torch.nn.functional.relu(x)
            x = self.sin_activation(x)
            
            if layer > 0:

                embed_Feat = embeddings_list[layer-1][:,:-self.max_points_per_level] + x
                # Style Modulation #
                #self.styleTransferBlock(x,embed_Feat)
                if self.has_out:
                    # for SIREN BRANCH
                    out_layer = getattr(self,"out_lin" + str(layer-1)).to(input.device)
                    x_high = out_layer(embed_Feat)
                    x_high = self.out_activation(x_high)

                    x_out = x_out + x_high
                else:
                    features_list.append(embed_Feat)
       

        if self.has_out:
            x = x_out
        else:
            features_list = torch.stack(features_list,dim=0).to(device=input.device)
            k = torch.zeros(x.shape[0],self.embeddings_dim-self.num_inputs,device=input.device)
            for i in range(0,self.n_ffenc_layers - 2):
                k += features_list[i]
            if self.include_input:
                x = torch.cat([input,k],dim=-1)
        out_feat = x
        out = out_feat/self.grid_levels
        return out

    """Functions Used for SIREN Layers"""
    def init_SIREN(self):
        for layer in range(0, self.n_ffenc_layers-1):
            lin = getattr(self, "ff_lin" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0)
    def init_SIREN_high(self):
        for layer in range(0, self.n_ffenc_layers-1):
            lin = getattr(self, "ff_lin" + str(layer)) 
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin,self.sin_w0_high)

    def init_SIREN_out(self):
        for layer in range(0, self.grid_levels):
            lin = getattr(self, "out_lin" + str(layer))

            sine_init(lin, w0=self.sin_w0_high)

    
        
    # optimizer utils
    def get_optimizer(self,lr,weight_decay):
        return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)  