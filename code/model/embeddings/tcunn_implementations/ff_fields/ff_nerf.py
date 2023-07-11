import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import tinycudann as tcnn
from model.custom_embeder_decoder import get_embedder as custom_embeder_decoder,Decoder





class NerfFF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False,
                 embed_type:str='FourierFeatures',
                 log2_max_hash_size = 16,
                 max_points_per_entry = 2,
                 base_resolution = 64,
                 desired_resolution = 1024):
        super(NerfFF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.multires = multires
        self.multires_view = multires_view
        self.output_ch = output_ch
        self.input_ch_view = 3
        self.skips = skips
        self.embed_fn = None
        self.embed_fn_view = None
        
        if multires > 0:
            embed_fn, input_ch = custom_embeder_decoder(input_dims=d_in, embed_type=embed_type, multires=multires,desired_resolution=512)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if  self.multires_view > 0:
            embed_fn_view, input_ch_view = custom_embeder_decoder(input_dims=d_in, embed_type=embed_type, multires=self.multires_view,desired_resolution=512)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view
        
        self.use_viewdirs = use_viewdirs
        self.nerf_config = {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "Softplus",
                "n_hidden_layers": self.D-2,
                "n_neurons": self.W
        }
        
        self.n_layers = self.D
        for i in range(self.D-1):
            if i not in self.skips:
                  self.pts_linears = tcnn.Network(self.W,self.W,self.nerf_config)
            else:
                  self.pts_linears = tcnn.Network(self.W+self.input_ch,self.W,self.nerf_config)
            
        self.view_linears = tcnn.Network(self.W+self.input_ch_view,self.W//2,self.nerf_config)
        
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W,W)
            self.alpha_linear = nn.Linear(W,1)
            self.rgb_linear = nn.Linear(W // 2,3)
            
        else:
            self.output_linear = nn.Linear(W,output_ch)
            
            
            
        
    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)
            
        h = input_pts
        
        h = self.pts_linears(h)
        for i in range(self.n_layers-1):
            if i in self.skips:
                h = torch.cat([input_pts,h],dim=-1)
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature,input_views],dim=-1)
            
            # pass through cutlass layers
            h = self.view_linears(h)
            
            rgb = self.rgb_linear(h)
            return alpha,rgb
        else:
            assert False, "Not Nerf without viewdirs implemented"
        
        
        


            
        