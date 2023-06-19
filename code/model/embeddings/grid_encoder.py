"""
    Initial attempt to implement
    MultiRes Hash Grid Encoding
    TOBE Ommit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class _GridMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_feats: int,
                 num_levels: int, max_points_per_level: int, base_resolution: float,
                 log2_hashmap_size: int, desired_resolution: float, grid_type:str = 'learned',
                 align_corners: bool = False, interpolation: str = 'linear'):
        
        super(_GridMLP,self).__init__()
        
        
        '''Initialize MLP hyper parameters'''
        self.layers = nn.ModuleList()
        input_size = input_dim
        for hidden_size in hidden_dims:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())
            input_size = hidden_size
            self.layers.append(nn.Linear(input_size, num_feats))

        '''Initialize grid encoding parameters'''
        self.num_levels = num_levels
        self.max_points_per_level = max_points_per_level
        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.desired_resolution = desired_resolution
        self.grid_type = grid_type
        self.align_corners = align_corners
        self.interpolation = interpolation


        '''Initialize Parameteres required for the spatial hashing function'''
        self.hash_prime_numbers = [2_654_435_761, 805_459_861, 1] + \
                          [12_345_677 * i + 1 for i in range(input_dim-3)]
        # Parameters required for the learned grid function
        self.num_features = num_feats
        self.hashmap_size = 2 ** (2 * log2_hashmap_size)
        self.voxel_size = self.base_resolution * (2 ** self.num_levels)
        self.hash_scale = self.hashmap_size / self.voxel_size
        
        if self.grid_type == 'learned':
            # Use learned parametrization of GridEncode Function
            # Initialize embedding parameters
            self.embedding_parameters = nn.Parameter(torch.Tensor(self.num_features, self.max_points_per_level))
            nn.init.kaiming_uniform_(self.embedding_parameters, a=5**0.5)
        elif self.grid_type == 'fixed':
            # Used fixed grid encode function
            print("Fixed GridEncodeFunction is selected")
        else:
            raise ValueError("Invalid grid_type: {}".format(grid_type))

    # Forward pass computed features from MLP layers
    def forward(self, input: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply MLP layers and then GridEncoderFunction
        tensor_input = F.relu(input)
        # Apply GridEncodeFunction to compute the spatial embedding of the features
        features = GridEncodeFunction.apply(coords,tensor_input,self.num_levels,self.max_points_per_level,
                                            self.base_resolution,self.log2_hashmap_size,self.desired_resolution,
                                            self.grid_type, self.align_corners, self.interpolation)
        mlp_output = features
        for layer in self.layers:
            mlp_output = F.relu(layer(mlp_output))

        return mlp_output

    def backward(self,grad_output):
        # Backward pass through the GridEncodeFunction
        grad_input = GridEncodeFunction.backward(grad_output)
        return grad_input
    def loss(self, input_coords, target_coords):
        # Compute loss between input and target coordinates
        loss = torch.nn.MSELoss()
        return loss(input_coords, target_coords)


        
        
        
class GridEncodeFunction(torch.autograd.Function):
    def hash_function( x: torch.Tensor, T:int) -> torch.Tensor:
        """
            Spatial hashing function as described in instant-ngp paper
            
            Args: 
            x (torch.Tensor): Input coordinate of shape (batch_size,num_points, input_dim).
            T (int): hashmap size.
            
            Return:
            hash(torch.Tensor): Hashed value of shape (batch_size, num_points, input_dim).
        
        """
        d = x.shape[-1]
        
        
        prime_nums =  [2_654_435_761, 805_459_861, 1] + \
                          [12_345_677 * i + 1 for i in range(d-3)]
        
        h_lookup = []
        """
            Xor operation is a bitwise operation.
            The code applies the hash operation explained above by performing the per-dimension linear congruential (pseudo-random) permutation as follows:

            For each dimension 𝑖=1, 2, ..., 𝑑 of the input coordinate tensor x, 
            the code multiplies the value of x along that dimension by the corresponding prime number 𝜋𝑖 and creates a tensor h_i of the same shape.
            
            The code concatenates the tensors h_i along the last dimension using torch.cat to obtain a final tensor h of shape (*x.shape[:-1], d), 
            where d is the number of dimensions and *x.shape[:-1] corresponds to all the dimensions of x except the last one.

            The resulting tensor h is the hashed tensor of the input coordinate x that will be used to index into the feature vector array.
        """
        
        for i in range(d):
            h_i = x[..., i:i+1] * torch.tensor(prime_nums[i], device = x.device)
            h_lookup.apppend(h_i)
            
        h  = torch.cat(h_lookup, dim = -1)
        
        hash = h.sum(dim=-1).long() % T
        
        return hash 
    @staticmethod
    def apply(self,ctx, coords, feats, num_levels, max_points_per_level,
                base_resolution,log2_hashmap_size, desired_resolution, gridtype,
                align_corners, interpolation):
        
        # Calculate the level growth factor 
        beta_growth =  torch.exp((torch.log(desired_resolution) - torch.log(base_resolution)) / (num_levels - 1))
        
        # Calculate the i-th level resolution table 
        per_level_res = [base_resolution * (beta_growth ** i) for i in range(num_levels)]
        D = coords.shape[-1]
        # Calculate the max_enteries for each level 
        max_entries_per_level = [int(math.ceil(max_points_per_level * (per_level_res[i] ** D))) for i in range(num_levels)]
        
        
        # Initialize the feature vector array -
        features = []
        features = torch.zeros((num_levels,max_entries_per_level, feats))


        # Apply hash encoding to each level
        hash_factors = [torch.tensor([1] + [int(torch.rand(1) * 2**32 - 1) for j in range(coords.shape[-1] - 1)]) for i in range(num_levels)]
        hash_table_size = 2 ** log2_hashmap_size
        for l in range(num_levels):
            level_resolution = per_level_res[l]
            level_features = features[l]

            # Scale coordinates by level resolution
            scaled_coords = coords * level_resolution

            # Round down and up
            floor_coords = torch.floor(scaled_coords)
            ceil_coords = torch.ceil(scaled_coords)

            # Calculate hash indices for each corner
            corner_coords = [floor_coords, ceil_coords]
            corner_indices = []
            corner_indices = self.hash_function(corner_coords, hash_table_size)
            
            # for corner_coord in corner_coords:
            #     corner_index = torch.zeros(corner_coord.shape[:-1], dtype=torch.int64)
            #     for d in range(coords.shape[-1]):
            #         corner_index += (corner_coord[..., d] * hash_factors[l][d]) % hash_table_size
            #     corner_indices.append(corner_index)
            # corner_indices = torch.stack(corner_indices)

            # Add feature vector to hash table
            for i in range(coords.shape[-1] ** 2):
                flat_indices = corner_indices.reshape(2, -1)[:, i]
                flat_features = level_features[flat_indices]
                flat_weights = torch.clamp((scaled_coords - floor_coords).reshape(2, -1)[:, i], 0, 1)
                interp_weights = torch.prod(flat_weights, dim=0)
                level_features[flat_indices[0]] += interp_weights[0] * flat_features[0]
                level_features[flat_indices[1]] += interp_weights[1] * flat_features[1]

        # Save necessary variables for backward pass
            ctx.save_for_backward(coords, feats, num_levels, max_points_per_level, base_resolution,
                                log2_hashmap_size, desired_resolution, gridtype, align_corners, interpolation,
                                beta_growth, per_level_res, max_entries_per_level, hash_factors, hash_table_size)

            return features
        
    @staticmethod
    def backward(ctx, grad_output):
        # Unpack the saved tensors
        (coords, feats, num_levels, max_points_per_level, _, _,
        _, _, _, _, _, per_level_res,
        _, hash_factors, hash_table_size, weight, bias) = ctx.saved_tensors

        # Allocate tensors for gradients
        grad_coords = torch.zeros_like(coords)
        grad_feats = torch.zeros_like(feats)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        # Compute gradients with respect to feature vectors in hash table
        for l in range(num_levels):
            level_resolution = per_level_res[l]
            level_features = ctx.saved_tensors[1][l]

            # Scale coordinates by level resolution
            scaled_coords = coords * level_resolution

            # Round down and up
            floor_coords = torch.floor(scaled_coords)
            ceil_coords = torch.ceil(scaled_coords)

            # Calculate hash indices for each corner
            corner_coords = [floor_coords, ceil_coords]
            corner_indices = []
            for corner_coord in corner_coords:
                corner_index = torch.zeros(corner_coord.shape[:-1], dtype=torch.int64)
                for d in range(coords.shape[-1]):
                    corner_index += (corner_coord[..., d] * hash_factors[l][d]) % hash_table_size
                corner_indices.append(corner_index)
            corner_indices = torch.stack(corner_indices)

            # Compute weights for each corner essentially is the scaled_coords - floor_coords clamped to [0, 1] and the computing the product we get the interpolation
            # of the weights
            flat_weights = torch.clamp((scaled_coords - floor_coords).reshape(2, -1), 0, 1)
            interp_weights = torch.prod(flat_weights, dim=0)  # (2 * max_points_per_level)

            # Compute gradients for each corner and accumulate to feature vectors and weights
            for i in range(coords.shape[-1] ** 2):
                flat_indices = corner_indices.reshape(2, -1)[:, i]
                flat_grads = grad_output[l, flat_indices] * interp_weights[i]
                grad_feats[l, flat_indices[0]] += flat_grads[0]
                grad_feats[l, flat_indices[1]] += flat_grads[1]
                grad_weight[l] += flat_grads[0] * level_features[i]
                grad_weight[l] += flat_grads[1] * level_features[i+max_points_per_level]

        # Compute gradients with respect to input coordinates
        for l in range(num_levels):
            level_resolution = per_level_res[l]
            level_features = ctx.saved_tensors[1][l]

            # Scale coordinates by level resolution
            scaled_coords = coords * level_resolution

            # Round down and up
            floor_coords = torch.floor(scaled_coords)
            ceil_coords = torch.ceil(scaled_coords)

            # Calculate hash indices for each corner
            corner_coords = [floor_coords, ceil_coords]
            corner_indices = []
            for corner_coord in corner_coords:
                corner_index = torch.zeros(corner_coord.shape[:-1], dtype=torch.int64)
                for d in range(coords.shape[-1]):
                    corner_index += (corner_coord[..., d] * hash_factors[l][d]) % hash_table_size
                corner_indices.append(corner_index)
            corner_indices = torch.stack(corner_indices)

            # Compute weights for each corner
            flat_weights = torch.clamp((scaled_coords - floor_coords).reshape(2, -1), 0, 1)
            interp_weights = torch.prod(flat_weights, dim=0)

            # Compute gradients for each corner and accumulate to feature vectors and weights
            for i in range(coords.shape[-1] ** 2):
                flat_indices = corner_indices.reshape(2, -1)[:, i]
                flat_grads = grad_output[l, flat_indices] * interp_weights[i]
                grad_coords += flat_grads[0] * weight[l] * level_features[i]
                grad_coords += flat_grads[1] * weight[l] * level_features[i+max_points_per_level]


            return grad_coords, grad_feats, grad_weight, grad_bias


                
if __name__ == '__main__':
    input = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    embeds = _GridMLP(input_dim=input.shape[0], hidden_dims=[input.shape[0],16,32,16], num_feats=16,
                                              num_levels=16, max_points_per_level= 2**14, base_resolution=16,
                                              log2_hashmap_size=14*torch.log(torch.tensor(2)), desired_resolution=512,
                                              align_corners=False, interpolation='linear')


    torch.set_printoptions(profile="full")
    print("embeds", embeds)
    