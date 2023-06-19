import torch
import torch.nn as nn

import tinycudann as tcnn 
import math 

from ..fourierFeatureModels import FourierFeaturesMLP
from hashGridEncoderTcnn import MultiResHashGridEncoderTcnn as HashEncoderTcnn




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
        