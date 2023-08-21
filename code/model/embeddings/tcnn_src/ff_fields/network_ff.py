import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import tinycudann as tcnn
from model.custom_embeder_decoder import get_embedder as custom_embeder_decoder,Decoder

    
class SDFNetworkFF(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=16,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 embed_type:str='Dummy',
                 log2_max_hash_size = 16,
                 max_points_per_entry = 2,
                 base_resolution = 64,
                 desired_resolution = 1024):
        super(SDFNetworkFF, self).__init__()
        
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.d_in = d_in
        self.embed_fn_fine = None
        self.embed_type = embed_type
        self.skip_in = skip_in
        self.multires = multires
        self.bias = bias
        self.scale = scale
        
        if multires > 0:
            print(self.embed_type)
            # create embedder model 
            embed_fn, input_ch = custom_embeder_decoder(self.d_in, self.embed_type, self.multires, desired_resolution)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
            
        self.n_layers = n_layers
        
        # Create template ff net config 
        ff_net_config = {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "Softplus",
                "weight_init": "Xavier",
                "bias_init": "Xavier",
                "weight_norm": "None",
                "weight_norm_axes": [],
                "dropout": 0.0,
                "batch_norm": False,
                "batch_norm_momentum": 0.1,
                "batch_norm_affine": True,
                "batch_norm_track_running_stats": True,
                "batch_norm_eps": 1e-5,
                "batch_norm_num_features": None,
                "n_hidden_layers": self.n_layers,
                "n_neurons": 256,
        }
        
        # create weight initialization vector for geometric init
        
        # Create 3D space geometric initialazation with sphere 
        for l in range(0, self.n_layers -1):
            if l+1 in self.skip_in:
                out_dim = dims[l+1] + dims[0]
            else:
                out_dim = dims[l+1]
                
            
        # Call it to initialize the field - sphere 
        def pre_train_sphere(self, iter):
            print ("Initialize SDF to sphere")
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

            for i in range(iter):
                p = torch.rand((1024,3), device='cuda') - 0.5
                ref_value  = torch.sqrt((p**2).sum(-1)) - 0.3
                output = self(p)
                loss = loss_fn(output[...,0], ref_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Pre-trained MLP", loss.item())
                
        
        # Create backbone network using TinyCudaNN FullyFusedMLP
        
        self.backbone = tcnn.Network(self.d_in,out_dim,ff_net_config)
        
    def sdf(self, input):
        return self.backbone(input)
    
    def sdf_hidden_appearence(self,input):
        return self.backbone(input)[:,1:]
    
    
    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        
        x = inputs
        
        for l in range(self.n_layers):
            x = self.backbone(x)
            
            if l in self.skip_in:
                x = torch.cat([x, inputs], dim=1) / np.sqrt(2)
            
            if l < self.n_layers - 2:
                x = F.relu(x)
        return torch.cat([x[:,:1] / self.scale, x[:,1:]], dim=-1)
    def gradient(self,x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
    