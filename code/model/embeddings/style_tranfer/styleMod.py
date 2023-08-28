import torch
import torch.nn as nn

from model.embeddings.style_tranfer.utils_function import *

# class StyleMod(nn.Module):
#     def __init__(self, feature_vector_size):
#         super(StyleMod, self).__init__()
#         # substruct input dims from feature_vector_size
#         self.style_fc = nn.Linear(feature_vector_size-3, 2).to(device='cuda')

#     def forward(self, content_feat, style_feat):
#         batch_size, num_channels= content_feat.size()
#         style_params = self.style_fc(style_feat)
#         style_scale = style_params[:, 0].unsqueeze(1).expand_as(content_feat)
#         style_shift = style_params[:, 1].unsqueeze(1).expand_as(content_feat)
#         normalized_feat = adaptive_instance_normalization(content_feat, style_feat)
#         styled_feat = normalized_feat * style_scale + style_shift
#         return styled_feat

""" Alternative StyleBlock Implementation """ 
class StyleMod(torch.nn.Module):
    def __init__(self, in_channels, style_dim):
        super(StyleMod, self).__init__()
        self.fc = torch.nn.Linear(style_dim, in_channels).to(device='cuda')
        self.bias = torch.nn.Parameter(torch.zeros(1, in_channels, 1, 1))
       
    def forward(self, x, style):
        batch_size = x.shape[0]
        # Compute style modulation parameters
        style = self.fc(style)
        scale, bias = style.chunk(2, dim=1)
        
        # Weight demodulation
        mean = torch.mean(x, dim=[0, 1], keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x ** 2, dim=[0, 1], keepdim=True) + 1e-8)
        x = x / std
        
        # Modulation
        x = x * scale + bias
        
        # Convolution and bias addition
        x = x + self.bias
        
        return x
    
