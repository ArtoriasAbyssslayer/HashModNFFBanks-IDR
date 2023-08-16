import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd
import tinycudann as tcnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MultiResHashGridEncoderTcnn(nn.Module):
    def __init__(self, 
                 include_input: bool,
                 in_dim: int,
                 network_dims: list,
                 embed_type: str,
                 n_levels: int,
                 max_points_per_level: int,
                 log2_hashmap_size: int,
                 base_resolution: int,
                 desired_resolution: int,
                 base_sigma: float,
                 exp_sigma: float,
                 grid_embedding_std: float,
                 per_level_scale: float):
        
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input
        self.embed_type = embed_type
        self.n_levels = n_levels
        self.max_points_per_level = max_points_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        for p in self.parameters():
            p.requires_grad = False
        
        # --- Hash Primes ---
        
        
        # --- Initialize Grid Encoder ---
        
        """
            {
                "otype": "Grid",           // Component type.
                
                "type": "Hash",            // Type of backing storage of the
                                           // grids. Can be "Hash", "Tiled"
                                           // or "Dense".
                
                "n_levels": 16,            // Number of levels (resolutions)
                
                "n_features_per_level": 2, // Dimensionality of feature vector
                                           // stored in each level's entries.
                
                "log2_hashmap_size": 19,   // If type is "Hash", is the base-2
                                           // logarithm of the number of elements
                                           // in each backing hash table.
                
                "base_resolution": 16,     // The resolution of the coarsest le-
                                           // vel is base_resolution^input_dims.
                
                "per_level_scale": 2.0,    // The geometric growth factor, i.e.
                                           // the factor by which the resolution
                                           // of each grid is larger (per axis)
                                           // than that of the preceding level.
                
                "interpolation": "Linear"  // How to interpolate nearby grid
                                           // lookups. Can be "Nearest", "Linear",
                                           // or "Smoothstep" (for smooth deri-
                                           // vatives).
            }
        """
        if self.embed_type == 'HashGridTcnn':
                otype = "Grid"
                type = "Hash"
        self.grid_encoder = tcnn.Encoding(
            n_input_dims=self.in_dim,
            encoding_config={
                "otype": otype,
                "type": type,
                "n_levels": int(self.n_levels),
                "n_features_per_level": self.max_points_per_level,
                'log2_hashmap_size': self.log2_hashmap_size,
                'base_resolution': self.base_resolution,
                'per_level_scale': self.per_level_scale,
                'exp_sigma': exp_sigma,
                'base_sigma':base_sigma,
                'hidden_dims': network_dims,
                'grid_embedding_std': grid_embedding_std,
                "interpolation": "Linear"
                
            }
        )
        self.grid_levels = self.n_levels
        print(f"Selected Grid Encoder Levels: {self.grid_levels}")
        if(self.include_input == True):
                self.output_dim = self.n_levels * self.max_points_per_level 
                self.embeddings_dim = self.in_dim + self.output_dim
        else:
            self.output_dim = self.n_levels * self.max_points_per_level
            self.embeddings_dim = self.output_dim    
        for p in self.grid_encoder.parameters():
            p.requires_grad = False
    def forward(self,x):
        torch.cuda.empty_cache()
        if self.include_input == True:
            return torch.cat([x,self.grid_encoder(x)],dim=-1)
        else:
            return self.grid_encoder(x)