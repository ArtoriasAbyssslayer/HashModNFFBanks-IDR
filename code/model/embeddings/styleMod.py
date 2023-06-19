import torch


"""
    This class will implement StyleGAN2 styleblock modulation and demodulation 
    to feature maps using weight demodulation and modulation.
    
    The feature Maps that are passed to the class are generated 
    by the embedding models.
    
    styleMod model will be used as in the following snippet:
    
"""

import torch

class StyleMod(torch.nn.Module):
    def __init__(self, in_channels, style_dim):
        super(StyleMod, self).__init__()
        self.fc = torch.nn.Linear(style_dim, in_channels * 2)
        self.bias = torch.nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        
    def forward(self, x, style):
        batch_size = x.shape[0]
        
        # Compute style modulation parameters
        style = self.fc(style).view(batch_size, -1, 1, 1)
        scale, bias = style.chunk(2, dim=1)
        
        # Weight demodulation
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) + 1e-8)
        x = x / std
        
        # Modulation
        x = x * scale + bias
        
        # Convolution and bias addition
        x = x + self.bias
        
        return x



