import torch
import torch.nn as nn

from model.embeddings.stylemod.utils_function import *

class StyleMod(nn.Module):
    def __init__(self, feature_vector_size):
        super(StyleMod, self).__init__()
        self.style_fc = nn.Linear(feature_vector_size, 2)

    def forward(self, content_feat, style_feat):
        batch_size, num_channels, height, width = content_feat.size()
        
        style_params = self.style_fc(style_feat)
        style_scale = style_params[:, 0].view(batch_size, num_channels, 1, 1)
        style_shift = style_params[:, 1].view(batch_size, num_channels, 1, 1)

        normalized_feat = adaptive_instance_normalization(content_feat, style_feat)
        styled_feat = normalized_feat * style_scale + style_shift

        return styled_feat

# Example usage
feature_vector_size = 256
style_mod = StyleMod(feature_vector_size)

# Assuming you have content_feat and style_feat as your input features
modified_feat = style_mod(content_feat, style_feat_selected)

#class StyleMod(torch.nn.Module):
    # def __init__(self, in_channels, style_dim):
    #     super(StyleMod, self).__init__()
    #     self.fc = torch.nn.Linear(style_dim, in_channels * 2)
    #     self.bias = torch.nn.Parameter(torch.zeros(1, in_channels, 1, 1))
       
    # def forward(self, x, style):
    #     batch_size = x.shape[0]
        
    #     # Compute style modulation parameters
    #     style = self.fc(style).view(batch_size, -1, 1, 1)
    #     scale, bias = style.chunk(2, dim=1)
        
    #     # Weight demodulation
    #     mean = torch.mean(x, dim=[2, 3], keepdim=True)
    #     x = x - mean
    #     std = torch.sqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) + 1e-8)
    #     x = x / std
        
    #     # Modulation
    #     x = x * scale + bias
        
    #     # Convolution and bias addition
    #     x = x + self.bias
        
    #     return x
    
