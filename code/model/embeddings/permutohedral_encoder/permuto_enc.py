import torch 
import permutohedral_encoding as permuto_enc 
import numpy as np 


# Create encoding 
class PermutohedralEncoder(torch.nn.Module):
    def __init__(self, input_dims=3, num_channels=128, include_input=True):
        super(PermutohedralEncoder, self).__init__()
        self.pos_dim = input_dims
        self.capacity = 4 * num_channels
        self.nr_levels = 8
        self.nr_feat_per_level = 2
        self.coarsest_scale = 1.0
        self.finest_scale = 0.00001
        self.scale_list = np.geomspace(self.coarsest_scale, self.finest_scale, num=self.nr_levels)
        self.include_input = include_input
        self.embeddings_dim = (2 * self.nr_levels + input_dims)  if include_input else 2 * self.nr_levels

    def forward(self, inputs):
        encoding = permuto_enc.PermutoEncoding(self.pos_dim, self.capacity, self.nr_levels, self.nr_feat_per_level, self.scale_list)
        permuto_embeds = encoding(inputs)
        
        if self.include_input:
            return torch.cat([inputs, permuto_embeds], -1)
        else:
            return permuto_embeds