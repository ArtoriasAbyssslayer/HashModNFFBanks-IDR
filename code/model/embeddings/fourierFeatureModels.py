import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Optional

'''
    Implementation of Fourier Feature Net based on Fourier Features Paper \href: https://arxiv.org/abs/2006.10739
    and Github repository: https://github.com/matajoh/fourier_feature_nets/blob/main/fourier_feature_nets/fourier_feature_models.py
'''


class FourierFeaturesMLP(nn.Module):
    """MLP which uses Fourier features as embedding preproc step"""
    # use optional parameters because in some cases they are not expected
    def __init__(self,include_input:bool, d_in: int, d_out: int,
                 a_vals: Optional[torch.Tensor]=None, b_vals: Optional[torch.Tensor]=None,
                 layer_channels: Optional[List[int]]=None):
        """Fourier Embedding MLP Constructor
            Args:
                d_in(int): Number of Dimensions of the input layer
                d_out(int): Embedding output layer outputs
                a_vals(torch.Tensor): a values in the fourier feature trans defining the scaling coefficient of each sinusoidal component
                b_vals(torch.Tensor): b values in the fourier feature trans defining the harmonic freq of each sinusoidal component
                layer_channels(List): List of integer defining the number of neurons(feature - dimensionality) in each layer
        """
        # Pass the params to the nn.Module
        super(FourierFeaturesMLP,self).__init__()
        self.params = {
            "include_input": include_input,
            "d_in": d_in,
            "d_out": d_out,
            "a_vals": None if a_vals is None else a_vals.tolist(),
            "b_vals": None if b_vals is None else b_vals.tolist(),
            "layer_channels": layer_channels
        }
        if b_vals is None:
            self.a_vals = None
            self.b_vals = None
        else:
            assert b_vals.shape[0] == d_in
            assert a_vals.shape[0] == b_vals.shape[0]
            
            d_in = b_vals.shape[1] * 2
            
        # Define the mlp based on embedding tensor shape and pass through the layers the input
        d_layers = []
        d_layers.append(nn.Linear(d_in, layer_channels[0]))
        for num_channels in self.params['layer_channels']:
            d_layers.append(nn.Linear(num_channels, num_channels))
        
        d_layers.append(nn.Linear(d_in, self.params['d_out']))
        self.d_layers = nn.Sequential(*d_layers)
        self.embeddings_dim =  self.params['d_out']
        # self.activations = []
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predicts the outputs from the provided uv input."""
        if self.b_vals is None:
            output = inputs
        else:
            embed_fqs = (self.a_vals*inputs * math.pi)@self.b_vals
            print(embed_fqs.shape)
            output = torch.cat([embed_fqs.cos(), embed_fqs.sin()], dim=0)
            output = output[:,None]
        # self.activations.append(F.relu() for _ in range(len(self.d_layers)))         # Make the forward pass from a relu function
        for layer in self.d_layers[:-1]:
            output = F.relu(layer(output))
        output = self.d_layers[-1](output)
        
        output = output.reshape(1, self.params['d_out']
                                )
        if self.params['include_input']:
            output_dim = output.shape[-1] + inputs.shape[-1]
            
            output = torch.cat([output, inputs], dim=-1)
            self.embeddings_dim = output_dim
        else:
            self.embeddings_dim = output.shape[-1]
            
        
       
        return output.reshape(1, self.embeddings_dim)

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


