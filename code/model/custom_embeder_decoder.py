import torch
import torch.nn as nn
import numpy as np 
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
from model.embeddings.fourierFeatureModels import *
from model.embeddings.fourierFilterBanks import FourierFilterBanksMLP
from model.embeddings.tcunn_implementations.hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as MRHashGridEncTcnn
"Define Embedding model selection function and Network Object Initialization"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_embedder(input_dims, embed_type, multires,log2_max_hash_size,max_points_per_entry,mapping_size,base_resolution,desired_resolution):
    
    """
        This class is responsible for selecting the embedding model and initializing the network object
        The Fourier Features Embendding Models (FourierFeaturesMLP,GaussianMLP,PositionalFourierMLP) are initialized with the same parameters
        
        Args:
        
            input: (Optional) include Input for having some input 
    """
    embed_kwargs = {
        'multi_resolution': {
            'include_input': True,
            'in_dim': input_dims,
            # 'max_freq_log2': multires-1,
            'n_levels':multires,
            'max_points_per_level': max_points_per_entry,
            'log2_hashmap_size': log2_max_hash_size,
            'base_resolution': base_resolution,
            'desired_resolution': desired_resolution            
            
        },
        'fourier_filter_banks':{
            'include_input':True,
            'in_dim': input_dims,
            'num_outputs': desired_resolution,
            'layer_channels':multires,
            'num_freqs': 5,
        },
        'fourier_mlp': {
            'include_input': True,
            'd_in': input_dims,
            'd_out': mapping_size,
            'a_vals': torch.from_numpy(1 / np.arange(1,input_dims+1).astype(np.float32)).to(device=DEVICE),
            'b_vals': torch.from_numpy(torch.randn((input_dims,mapping_size)).cpu().numpy().astype(np.float32)).to(device=DEVICE),
            'layer_channels': [mapping_size for _ in range(multires)]
        },
        'hashGridEncoderTcnn':{
            'include_input':True,
            'in_dim': input_dims,
            'embed_type': embed_type,
            'n_levels': multires,
            'max_points_per_level': max_points_per_entry,
            'log2_hashmap_size': log2_max_hash_size,
            'base_resolution': base_resolution,
            'per_level_scale': 2.0,
        },
    }
    
    
    embed_models = {
        'HashGrid': (MultiResHashGridMLP, 'multi_resolution'),
        'FFB': (FourierFilterBanksMLP, 'fourier_filter_banks'),
        'FourierFeatures': (FourierFeaturesMLP, 'fourier_mlp'),
        'GaussianFourier': (GaussianFourierMLP, 'fourier_mlp'),
        'PositionalEncoding':(PositionalFourierMLP,'forurier_mlp'),
        'HashGridTcnn':(MRHashGridEncTcnn,'hashGridEncoderTcnn'),
    }
    
    if embed_type not in embed_models:
        raise ValueError("Not a valid embedding model type")
    EmbedderClass, model_key = embed_models[embed_type]
    selected_kwargs = embed_kwargs[model_key]
    
    embedder_obj = EmbedderClass(**selected_kwargs)
    # Apply Embedding to the Input
    def embed(x, eo=embedder_obj): 
        return eo(x)
    
    return embed, embedder_obj.embeddings_dim

# Utility for geometric initialization of MLP - To be used for pre-training the sdf layers - Imlicit Rendering Network / Renderer / 
# MLP + Positional Encoding
class Decoder(torch.nn.Module):
    def __init__(self, input_dims = 3, internal_dims = 128, output_dims = 4, hidden = 5, multires = 2,desired_resolution=64, embed_type = 'PositionalEncoding'):
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(input_dims, embed_type, multires,internal_dims,desired_resolution)
            self.embed_fn = embed_fn
            input_dims = input_ch

        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, p):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        out = self.net(p)
        return out

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