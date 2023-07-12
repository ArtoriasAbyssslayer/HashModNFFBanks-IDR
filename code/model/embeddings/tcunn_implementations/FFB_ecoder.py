import torch
import torch.nn as nn
import tinycudann as tcnn 
import math 

from model.embeddings.fourier_encoding import FourierEncoding
from hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as HashEncoderTcnn
from .Sine import Sine,sine_init,first_layer_sine_init




class FFB_encoder(nn.Module):
    def __init__(self,
                 in_dim:int,
                 mlp_network_config,
                 encoding_config,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.mlp_network_config = mlp_network_config
        self.encoding_config = encoding_config
        