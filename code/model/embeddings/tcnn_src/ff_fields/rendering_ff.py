import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import tinycudann as tcnn
from model.custom_embeder_decoder import get_embedder as custom_embeder_decoder,Decoder



class RenderingNetworkFF(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=4,
                 squeeze_out=True,
                 embed_type:str='FourierFeatures',
                 log2_max_hash_size = 16,
                 max_points_per_entry = 2,
                 base_resolution = 64,
                 desired_resolution = 1024):
        
        super(RenderingNetworkFF, self).__init__()
        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in+ d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = custom_embeder_decoder(input_dims=d_in,embed_type=embed_type, multires=multires_view,desired_resolution=desired_resolution)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch+3
        
        self.num_layers =  len(dims)
        # Pass parameters to the nn.Module object
        self.n_layers = n_layers
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.d_feature = d_feature
        self.weight_norm = weight_norm
        self.multires_view = multires_view
        self.embed_type = embed_type
        self.log2_max_hash_size = log2_max_hash_size
        self.max_points_per_entry = max_points_per_entry
        self.base_resolution = base_resolution
        self.desired_resolution = desired_resolution    

        # Create ff net config 
        ff_net_config = {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "Softplus",
                "n_hidden_layers": self.n_layers,
                "n_neurons": 256,
        }
        
        
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l+1]
            if weight_norm:
                ff_net_config[f"weight_norm_{l}"] = "SpectralNorm"
                ff_net_config[f"weight_norm_{l}_dim"] = 0
    
        
        # Assign tcnn.Network model with its parameters
        self.backbone = tcnn.Network(self.d_in,self.d_out,ff_net_config)
    # Create forward Function
    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        rendering_input = None
        
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        
        x = rendering_input
        
        # Pass the input tHrough the cutlass network - backbone
        
        ff_net_config = {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "Softplus",
                "n_hidden_layers": self.n_layers,
                "n_neurons": 256,
        }
        
        if self.squeeze_out:
            ff_net_config[f"output_activation"] = "Sigmoid"
            self.backbone = tcnn.Network(self.d_in,self.d_out,ff_net_config)
            self.backbone(x)
        else:
            self.backbone(x)
        
        
            