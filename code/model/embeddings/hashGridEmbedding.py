# HashEmbedding Class

import numpy as np
import torch
import torch.nn as nn 
# Freuqency Encoding the auxiliary input features
from model.embeddings.frequency_enc import FourierFeature as FrequencyEncoding
"""
    Code Based on Ending Hsiao work on hashGridEmbedding image Features based on instant-ngp hashGridEncoding
    https://github.com/Ending2015a/hash-grid-encoding 
    The code has been modified based on the instant-ngp paper as I interpreted it 
"""
# ---- constants
HASH_PRIMES = [1,2654435761,805459861,3674653429,2097192037,1434869437,2165219737]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# save tensor to gpu slow 
primes = torch.tensor(HASH_PRIMES).to(DEVICE)
# repeat the same array to increase the length of hash primes for large dimension inputs

# _get_primes is deprecated and not used because I used fixed primes array 
def _get_primes(d: torch.Tensor):
    if  d < len(HASH_PRIMES):
        return HASH_PRIMES[:d]
    else:
        repeated_hash_primes = torch.tile(HASH_PRIMES, (d - len(HASH_PRIMES), 1))
        stacked_array = repeated_hash_primes.flatten()
        return stacked_array



# ---Multi Resolution HashGrid - Spatial Encoding ---


@torch.no_grad()
def hash_func(idx: torch.Tensor, primes: torch.Tensor, hashmap_size: int):
    d = idx.shape[-1]
    idx = (idx * primes[:d]) & 0xffffffff  # uint32
    for i in range(1, d):
        # bitwise xor
        idx[..., 0] ^= idx[..., i]
    # apply hash based on grid size
    return idx[..., 0] % hashmap_size

class _HashGridMLP(nn.Module):
    """
        Single Resolution Grid HashEmbedding Net

        Inputs:
            - dim : Input dimensions supports <= len(PRIMES)
            - n_features: Lookup table hash value size - max_points_per_level
            - hashmap_size: Size of the grid 
            - resolution: HashGrid Resolution
        Return:
            - forward->interpolated hash ids embeddings of the Inputs
        Training achieved due to the weight - lookup table mapping that nn.Embedding does and bpp is applicable
    """

    def __init__(self,dim: int,n_features: int,hashmap_size: int,resolution: float):
        super(_HashGridMLP,self).__init__()
        self.dim = dim
        self.n_features = n_features
        self.hashmap_size = hashmap_size
        self.resolution = resolution
        
        # you can add more primes for supporting more dimensions
        assert self.dim <= len(HASH_PRIMES), \
        f"HashGrid only supports < {len(HASH_PRIMES)}-D inputs"

        # create look-up table
        embedding = nn.Embedding(hashmap_size, n_features)
        std = 1e-4
        # custom_kaiming_uniform_(embedding.weight, std = std, a=0)
        nn.init.uniform_(embedding.weight, -std, std) 
        self.embedding = embedding.to(DEVICE)
        self.primes = primes

        # create interpolation binary mask
        n_neigs = 1 << self.dim
        neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
        dims = np.arange(self.dim, dtype=np.int64).reshape((1, -1))
        bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool) # (neig, dim)
        self.register_buffer('bin_mask', bin_mask, persistent=False)
    def forward(self, x: torch.Tensor,compute_grad=False)->torch.Tensor:
        # x: (b..., dim), torch.float32, range: [0, 1]
        bdims = len(x.shape[:-1])
        x = x * self.resolution
        xi = x.long()
        xf = x - x.float()
        # to match the input batch shape unsqueeze 
        xi = xi.unsqueeze(dim=-2) # (b..., 1, dim)
        xf = xf.unsqueeze(dim=-2) # (b..., 1, dim)
        # to match the input batch shape use bin_mask 
        bin_mask = self.bin_mask.reshape((1,)*bdims + self.bin_mask.shape)# (1..., neig, dim)
        # get neighbors' indices and weights on each dim
        inds = torch.where(bin_mask, xi, xi+1)# (b..., neig, dim)
        ws = torch.where(bin_mask, 1-xf, xf)# (b...., neig, dim)
        # aggregate nehgibors' interp weights - attention mechanism 
        w = ws.prod(dim=-1, keepdim=True) # (b..., neig, 1)
        # hash neighbors' id and look up table
        hash_ids = hash_func(inds, self.primes, self.hashmap_size)# (b..., neig)
        neig_data = self.embedding(hash_ids)# (b..., neig, feat)

        # interpolate neighbors' data
        return torch.sum(neig_data * w, dim=-2) # (b..., feat)


class MultiResHashGridMLP(nn.Module):
    def __init__(
            self,
            include_input: bool,
            in_dim: int,
            n_levels: int,
            max_points_per_level: int,
            log2_hashmap_size: int,
            base_resolution: int,
            desired_resolution: int):
        """
            Based on Nvidia's hash grid encoding
            https://nvlabs.github.io/instant-ngp/

            The output of this Multi-Resolution Hash Embedding is obviously the n_levels * max_points_per_level
            which is the voxel size of the hash grid encoding
        """
        super(MultiResHashGridMLP,self).__init__()
        self.include_input = include_input
        self.in_dim = in_dim
        self.n_levels = n_levels
        self.max_points_per_level = max_points_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.desired_resolution = desired_resolution

        # from paper eq(3)
        import math
        beta_growth = math.exp((math.log(desired_resolution) - math.log(base_resolution)) / (base_resolution - 1))

        levels = []

        for level_idx in range(self.n_levels):
            resolution = math.floor(
                self.base_resolution * (beta_growth ** level_idx))
            #self.register_buffer('current_res', resolution, persistent=False)
            self.hashmap_size = min(
                resolution ** in_dim,
                2 ** log2_hashmap_size)
            levels.append(
                _HashGridMLP(
                    self.in_dim,
                    self.max_points_per_level,
                    self.hashmap_size,
                    resolution))
            self.levels = nn.Sequential(*levels).to(DEVICE)
            self.input_dim = in_dim
            if self.include_input == True:
                self.freq_encoding = FrequencyEncoding(in_dim,(math.log(desired_resolution) - math.log(base_resolution)) / (base_resolution - 1),num_channels=n_levels,include_input=True).to(DEVICE)  
            # Frequency Encoding Applied to Auxiliary Input which is concatenated to the HashGrid Embeddings
            if(include_input == True):
                self.output_dim = self.n_levels * self.max_points_per_level + (self.freq_encoding.embeddings_dim-self.input_dim)
                self.embeddings_dim = self.input_dim + self.output_dim
            else:
                self.embeddings_dim = self.output_dim
        
            
        for param in self.parameters():
            param.requires_grad = True
    def forward(self, x: torch.Tensor,compute_grad=False)->torch.Tensor:
        " In forard return concatenated emmbedding grids in each level resolution." 
        if self.include_input == True:
            return torch.cat([self.freq_encoding(x),torch.cat([level(x) for level in self.levels], dim=-1)],dim=-1)
        else:
            return torch.cat([level(x) for level in self.levels], dim=-1)
           
        




