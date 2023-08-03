import torch
import torch.nn as nn
import numpy as np 
from model.embeddings.hashGridEmbedding import MultiResHashGridMLP
from model.embeddings.fourier_encoding import FourierEncoding as FourierFeatures
from model.embeddings.nffb import FourierFilterBanks
#from model.embeddings.tcunn_implementations.hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as MRHashGridEncTcnn
#from model.embeddings.tcunn_implementations.FFB_encoder import FFB_encoder
from model.hash_encoder.hashgridencoder import MultiResolutionHashEncoderCUDA as MultiResHashGridEncoderCUDA 
"Define Embedding model selection function and Network Object Initialization"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Custom_Embedding_Network:
    """
        This class is responsible for selecting the embedding model and initializing the network object
        * The Fourier Features Embendding Models  are initialized with the same parameters as positional encoding
        * HashGrid parameters are initialized based on Nvidia's implementation of HashGrid Encoding (Neural Graphics Primitives)
        * The neural fourier filter banks models are initialized with the hashgrid parameters and the positional encoding parameters
    """
    def __init__(self,input_dims,network_dims,embed_type, multires,log2_max_hash_size,max_points_per_entry,base_resolution,desired_resolution,bound):
        embed_kwargs = {
            'MultiResHashEncoderCUDA':{
                'input_dim': input_dims,
                'num_levels': multires,
                'level_dim': max_points_per_entry,
                'per_level_scale': 2.0,
                'base_resolution': base_resolution,
                'log2_hashmap_size': log2_max_hash_size,
                'desired_resolution': desired_resolution,
            },
            'multi_resolution': {
                'include_input': True,
                'in_dim': input_dims,
                'n_levels':multires,
                'max_points_per_level': max_points_per_entry,
                'log2_hashmap_size': log2_max_hash_size,
                'base_resolution': base_resolution,
                'desired_resolution': desired_resolution            
                
            },
            'FFB_TCNN':{
                'HashGridEncoderConfig':{
                    'include_input':True,
                    'in_dim': input_dims,
                    'embed_type': 'HashGridTcnn',
                    'network_dims': network_dims,
                    'n_levels': multires,
                    'max_points_per_level': max_points_per_entry,
                    'log2_hashmap_size': log2_max_hash_size,
                    'base_resolution': base_resolution,
                    'desired_resolution': desired_resolution,
                    "base_sigma": 8.0,
                    "exp_sigma": 1.26,
                    "grid_embedding_std": 0.001,
                    'per_level_scale': 2.0,
                },
                'bound': bound,
            },
            'fourier_encoding': {
                'include_input': False,
                'input_dims': input_dims,
                'max_freq_log2': log2_max_hash_size,
                'num_freqs': multires,
                'log_sampling': True,
                'periodic_fns': [torch.sin, torch.cos]
            },
            'hashGridEncoderTcnn':{
                'include_input':True,
                'in_dim': input_dims,
                'embed_type': embed_type,
                'n_levels': multires,
                'max_points_per_level': max_points_per_entry,
                'log2_hashmap_size': log2_max_hash_size,
                'base_resolution': base_resolution,
                'desired_resolution': desired_resolution,
                "base_sigma": 8.0,
                "exp_sigma": 1.26,
                "grid_embedding_std": 0.001,
                'per_level_scale': 2.0,
            },
        }
        embed_models = {
            'HashGrid': (MultiResHashGridMLP, 'multi_resolution'),
            'FFB': (FourierFilterBanks, 'fourier_filter_banks'),
            'FourierFeatures': (FourierFeatures, 'fourier_encoding'),
            #'HashGridTcnn':(MRHashGridEncTcnn,'hashGridEncoderTcnn'),
            #'FFBTcnn':(FFB_encoder,'FFB_TCNN'),
            'HashGridCUDA': (MultiResHashGridEncoderCUDA, 'MultiResHashEncoderCUDA'),
        }   
        if embed_type not in embed_models:
            raise ValueError("Not a valid embedding model type")
        EmbedderClass, model_key = embed_models[embed_type]
        selected_kwargs = embed_kwargs[model_key]
        self.embedder_obj = EmbedderClass(**selected_kwargs)
        self.embeddings_dim = self.embedder_obj.embeddings_dim
    # Apply Embedding to the Input
    def embed(self,x): 
        return  self.embedder_obj(x)
    

# Utility for geometric initialization of MLP - To be used for pre-training the sdf layers - Imlicit Rendering Network / Renderer / 
# MLP + Positional Encoding
class Decoder(torch.nn.Module):
    def __init__(self, input_dims,internal_dims, output_dims, hidden,embed_fn,skip_in):
        super().__init__()
        self.embed_fn = embed_fn
        net = (torch.nn.Linear(input_dims, internal_dims[1], bias=False), torch.nn.ReLU())
        for i in range(2,hidden-2):
            if i in skip_in:
                output_dims = internal_dims[i+1] - internal_dims[0]
            else:
                output_dims = internal_dims[i+1]
            net = net + (torch.nn.Linear(internal_dims[i], internal_dims[i+1], bias=False), torch.nn.ReLU())
            
        net = net + (torch.nn.Linear(internal_dims[hidden-2], output_dims, bias=False),torch.nn.Tanh())
        self.net = torch.nn.Sequential(*net).to(DEVICE)

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