
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embeddings.style_Attention.style_function import *



"""
    
    The following Module applies StyleModulation 
    without AdaIN based on two specific feature vectors
    trying to modulate the one based on the other

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class StyleAttention(nn.Module):
    def __init__(self, feature_vector_size=28):
        super().__init__()
        self.feature_vector_size = feature_vector_size
        self.linear_transform = nn.Linear(feature_vector_size, feature_vector_size).to(device=device)
        self.attention = nn.Linear(feature_vector_size, 1).to(device=device)
        self.norm = nn.InstanceNorm1d(feature_vector_size)

    def forward(self, content, style):
        # No need to set the multires_levels
        
        # Content is the original 3D coordinate Vector
        content_features = content.view(-1, self.feature_vector_size)
        # Style is its embedding in the latent space of NFFB 
        style_features = style.view(-1, self.feature_vector_size)

        # No need to apply AdaIN
        modulated_features = self.linear_transform(style_features)

        attention_weights = self.attention(content_features)
        attention_weights = F.softmax(attention_weights, dim=1)

        weighted_features = attention_weights * modulated_features

        demodulated_features = self.norm(weighted_features)

        return demodulated_features

"""
    The following module applies StyleModulation
    with respect to the multires levels of the FourierGridFeatures
    applying AdaIN algorithm 
"""

class StyleModulation(nn.Module):
    def __init__(self, multires_levels=3, feature_vector_size=28):
        super().__init__()
        self.multires_levels = multires_levels
        self.feature_vector_size = feature_vector_size

        self.linear_transform = nn.Linear(feature_vector_size, feature_vector_size).to(device=device)
        self.attention = nn.Linear(feature_vector_size, 1).to(device=device)

        self.norm = nn.InstanceNorm1d(feature_vector_size)

    def forward(self, content, style):
        # Set the multires_levels to 1, since the content and style feature vectors are in a single resolution level

        content_features = content.view(-1, 3, content.shape[1])
        style_features = style.view(style.shape[1], self.multires_levels, self.feature_vector_size)

        # Apply AdaIN to the style features
        style_features = adaptive_instance_normalization(
            content_features, style_features)
        style_features = style_features.squeeze()
        modulated_features = self.linear_transform(style_features)

        attention_weights = self.attention(content_features).detach()
        attention_weights = F.softmax(attention_weights, dim=1)
        
        weighted_features = attention_weights * modulated_features

        demodulated_features = F.normalize(weighted_features, dim=1).squeeze(dim=0)

        return demodulated_features