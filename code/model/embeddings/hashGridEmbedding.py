# HashEmbedding Class

import numpy as np
import torch
import torch.nn as nn

"""
    Based on Ending Hsiao work on hashGridEmbedding image Features based on instant-ngp hashGridEncoding

    https://github.com/Ending2015a/hash-grid-encoding

"""


# ---- constants
HASH_PRIMES = [1,2654435761,805459861,3674653429,2097192037,1434869437,2165219737,4506014050,1234123232,454534535,14324134,151234151513,143135152134,5153415135,1234142124,1541351351]

# repeat the same array to increase the length of hash primes for large dimension inputs
def _get_primes(d: int):
    if  d < len(HASH_PRIMES):
        return HASH_PRIMES[:d]
    else:
        repeated_hash_primes = np.tile(HASH_PRIMES, (d - len(HASH_PRIMES), 1))
        stacked_array = repeated_hash_primes.flatten()
        return stacked_array


class Frequency(nn.Module):
    def __init__(self, dim: int, n_levels: int = 10):
        """
            Positional encoding from NeRF
            [sin(x),cos(x), sin(4x), cos(4x), sin(8x), cos(8x)
             .....,sin(2^n*x), cos(2^n*x)]

            Args:
                - dim(int) : input dimensions
                - n_levels(int,optional): frequency bands number
        """
        super().__init__()
        self.n_levels = n_levels
        assert self.n_levels > 0

        freqs = 2. ** torch.linspace(0., n_levels - 1, n_levels)
        self.register_buffer('freqs', freqs)

        # ---
        self.input_dim = dim
        self.output_dim = dim * n_levels * 2

    def forward(self, x: torch.Tensor):
        # unsqueeze in order to multiply it with freqs  space
        x = x.unsqueeze(dim=-1)  # (...,dim,1)
        x = x * self.freqs  # (..., dim,L)
        # concatenate the sin,cos input frequencies series
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1)  # (..,dim,L*2)
        return x.flatten(-2, -1)  # (...,dim*L*2) - flatten the tensor to 1D


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
            - in_dim : Input dimensions supports <= len(PRIMES)
            - feats: Lookup table hash value size
            - hashmap_size: Size of the grid
            - resolution: HashGrid Resolution
        Return:
            - forward->interpolated hash ids embeddings of the Inputs
        Training achieved due to the weight - lookup table mapping that nn.Embedding does and bpp is applicable
    """

    def __init__(
            self, in_dim: int, n_feats: int,
            hashmap_size: int, resolution: float):
        super().__init__()
        self.in_dim = in_dim
        self.n_feats = n_feats
        self.hashmap_size = hashmap_size
        self.resolution = resolution
        hash_pri = _get_primes(in_dim)
        assert self.in_dim <= len(hash_pri), \
            "Hashing is not supported for more than {len(PRIMES)} - input dimensions"
        # use nn.Embedding pytorch layer as a simple lookup table of hash
        # values
        self.embedding_layer = nn.Embedding(hashmap_size, n_feats)
        nn.init.kaiming_uniform_(self.embedding_layer.weight, a=5 ** 0.5)

        # initialize primes tensor
        primes = torch.tensor(hash_pri, dtype=torch.int64)
        self.register_buffer('primes', primes, persistent=False)

        # create interpolation binary mask

        # calculate the number of neighbors for each point in the source point
        # cloud.
        n_neighbors = 1 << self.in_dim
        print(n_neighbors)
        # create neighbors array 0 to neigbors-1
        neighbors = np.arange(n_neighbors, dtype=np.int32).reshape((-1, 1))
        # create dimenstion Level  array
        dims = np.arange(self.in_dim, dtype=np.int64).reshape((1, -1))

        bin_mask = torch.tensor(
            neighbors & (
                1 << dims) == 0,
            dtype=bool)  # (neig,dim)
        self.register_buffer('bin_mask', bin_mask, persistent=False)

    def forward(self, x: torch.Tensor):
        
        # x:(b...,dim), torch.float32, range:[0,1]
        base_res_dims = len(x.shape[:-1])
        
        x = x * self.resolution
        x_i = x.long()
        x_f = x - x_i.float().detach()
        xi = x_i.unsqueeze(dim=-2)  # (b...,1,dim)
        xf = x_f.unsqueeze(dim=-2)  # (b...,1,dim)

        # to match the input batch shape
        bin_mask = self.bin_mask.reshape(
            (1,) * base_res_dims + self.bin_mask.shape)  # (1..., neigbors,dim)
        # get neighbors' indices and weights on each dim
        indeces = torch.where(bin_mask, xi, xi + 1)  # (b...,neighbors,dim)
        weights = torch.where(bin_mask, 1 - xf, xf)  # //

        # calculate the weights product
        w = weights.prod(dim=-1, keepdim=True) # (b..., neig, 1)
        # hash neigbors' id and lookup table
        hash_ids = hash_func(indeces, self.primes, self.hashmap_size)
        neighbors_data = self.embedding_layer(hash_ids)  # (b...,neighbors)
        return torch.sum(neighbors_data * w, dim=-2)  # (b...,feat)


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
            which is the voxel size

            model.
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
                    in_dim=self.in_dim,
                    n_feats=self.max_points_per_level,
                    hashmap_size=self.hashmap_size,
                    resolution=resolution))
            self.levels = nn.ModuleList(levels)

            self.input_dim = in_dim
            
            if(include_input == True):
                self.output_dim = self.n_levels * self.max_points_per_level
                self.embeddings_dim = self.input_dim + self.output_dim
            else:
                self.output_dim = self.n_levels * self.max_points_per_level
                self.embeddings_dim = self.output_dim   
            
        # In forard return concatenated emmbedding grids in each level
        # resolution.
    def forward(self, x: torch.Tensor):
        if self.include_input == True:
            # print (" Hash Encoding Input")
            embed = []
            for level in self.levels:
                embed.append(level(x))
            embed = torch.cat(embed,dim=-1)
            # embed = torch.cat([level(x) for level in self.levels], dim=-1)
            return torch.cat([x, embed], dim=-1)
        else:
            return torch.cat([level(x) for level in self.levels], dim=-1)
           
        




