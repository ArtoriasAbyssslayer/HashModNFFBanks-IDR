import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Optional

'''
    Implementation of Fourier Feature Net based on Fourier Features Paper \href: https://arxiv.org/abs/2006.10739
    and Github repository: https://github.com/matajoh/fourier_feature_nets/blob/main/fourier_feature_nets/fourier_feature_models.py
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FourierFeaturesMLP(nn.Module):
    def __init__(self,include_input, d_in, d_out,a_vals: Optional[torch.Tensor], b_vals: Optional[torch.Tensor],num_hidden_layers, hidden_dim):
        
        
        """
            Fourier Embedding MLP Model
            Args:
                d_in(int): Number of Dimensions of the input layer
                d_out(int): Embedding output layer outputs
                a_vals(torch.Tensor): a values in the fourier feature trans defining the scaling coefficient of each sinusoidal component
                b_vals(torch.Tensor): b values in the fourier feature trans defining the harmonic freq of each sinusoidal component
                layer_channels(List): List of integer defining the number of neurons(feature - dimensionality) in each layer - This was not efficient so it was substituted with hidden_dim, num_hidden_layers
        """
        
        
        super(FourierFeaturesMLP, self).__init__()
        if a_vals is None or b_vals is None:
            self.a_vals = torch.randn(d_in).to(device)
            self.b_vals = torch.randn(d_in).to(device)
        else:
            self.a_vals = a_vals
            self.b_vals = b_vals
        self.include_input = include_input
        self.d_in = d_in
        self.d_out = d_out
        self.embeddings_dim =  d_in  # Each input dimension is embedded as cosine and sine components
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(d_in, hidden_dim).to(device=device))
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim).to(device=device))
        self.output_layer = nn.Linear(hidden_dim, d_out).to(device=device)
        
    def forward(self, inputs):
        # Compute Fourier features
        embed_fqs = (inputs * math.pi) @ self.b_vals
        embed_fqs = embed_fqs.transpose(0, 1)
        embeddings = torch.cat([self.a_vals @ embed_fqs.cos(), self.a_vals @ embed_fqs.sin()], dim=1)
        
        # Pass through hidden layers
        x = embeddings
        self.hidden_layers[0] = nn.Linear(self.d_in*embeddings.shape[1], self.hidden_layers[0].out_features).to(device=device)
        x = x.view(-1, self.d_in*embeddings.shape[1])
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        
        # Pass through output layer
        output = self.output_layer(x) 
        if self.include_input:
            output_dim = output.shape[-1] + inputs.shape[-1]
            
            samples_size = inputs.shape[0]
            
            # inputs_flattened = inputs.view(-1)
            outputs = output.unsqueeze(1).expand(-1,inputs.size(1),-1)
            inputs = inputs.unsqueeze_(1)
    
            outputs = outputs.transpose(0,2)
            inputs = inputs.transpose(1,2)
            output = torch.cat([outputs, inputs ], dim=0)
            output = output.squeeze(2)
            self.embeddings_dim =  output.shape[1]
            #print(output.shape)
        else:
            self.embeddings_dim = output.shape[-1]
        return output
    
    def save(self, path: str):
        """
            Save the current checkpoint of the model on
            path specified in the code

            Args:
                path (str): Path to the model encoded by pytorch
        """

        state_dict = self.state_dict()
        state_dict["type"] = "FourierFeatureNET"
        state_dict["params"] = self.params
        torch.save(state_dict, path)


class MLP(FourierFeaturesMLP):
    """Unencoded FFN, essentially a standard MLP."""

    def __init__(self, d_in: int, d_out: int, num_layers=3,
                 num_channels=256):
        """Constructor.
        Args:
            d_in (int): Number of dimensions in the input
            d_out (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
        """
        FourierFeaturesMLP.__init__(self, d_in, d_out,
                                   a_vals=None, b_vals=None,
                                   layer_channels=[num_channels]*num_layers)


class BasicFourierMLP(FourierFeaturesMLP):
    """Basic version of FFN in which inputs are projected onto the unit circle."""

    def __init__(self, d_in: int, d_out: int, num_layers=3,
                 num_channels=256):
        """Constructor.
        Args:
            d_in (int): Number of dimensions in the input
            d_out (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
        """
        a_values = torch.ones(d_in)
        b_values = torch.eye(d_in)
        FourierFeaturesMLP.__init__(self, d_in, d_out,
                                   a_values, b_values,
                                   [num_channels] * num_layers)


class PositionalFourierMLP(FourierFeaturesMLP):
    """Version of FFN with positional encoding."""
    def __init__(self, d_in: int, d_out: int, max_log_scale: float,
                 num_layers=3, num_channels=256, embedding_size=256):
        """Constructor.
        Args:
            d_in (int): Number of dimensions in the input
            d_out (int): Number of dimensions in the output
            max_log_scale (float): Maximum log scale for embedding
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            embedding_size (int, optional): The size of the feature embedding.
                                            Defaults to 256.
        """
        b_values = self._encoding(max_log_scale, embedding_size, d_in)
        a_values = torch.ones(b_values.shape[1])
        FourierFeaturesMLP.__init__(self, d_in, d_out,
                                   a_values, b_values,
                                   [num_channels] * num_layers)

    @staticmethod
    def _encoding(max_log_scale: float, embedding_size: int, d_in: int):
        """Produces the encoding b_values matrix."""
        embedding_size = embedding_size // d_in
        frequencies_matrix = 2. ** torch.linspace(0, max_log_scale, embedding_size)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(d_in) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, d_in)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix


class GaussianFourierMLP(FourierFeaturesMLP):
    """Version of a FFN using a full Gaussian matrix for encoding."""

    def __init__(self, d_in: int, d_out: int, sigma: float,
                 num_layers=3, num_channels=256, embedding_size=256):
        """Constructor.
        Args:
            d_in (int): Number of dimensions in the input
            d_out (int): Number of dimensions in the output
            sigma (float): Standard deviation of the Gaussian distribution
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            embedding_size (int, optional): Number of frequencies to use for
                                             the encoding. Defaults to 256.
        """
        b_values = torch.normal(0, sigma, size=(d_in, embedding_size))
        a_values = torch.ones(b_values.shape[1])
        FourierFeaturesMLP.__init__(self, d_in, d_out,
                                   a_values, b_values,
                                   [num_channels] * num_layers)
if __name__ == "__main__":
    # Code to test fourier features MLPs
    pass


