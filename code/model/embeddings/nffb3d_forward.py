# 3D shape fitting encoder based on nffb 

import torch 
import torch.nn as nn

from model.embeddings.nffb3d import FourierFilterBanks as FFB_encoder
from model.embeddings.Sine import sine_init
from model.embeddings.style_tranfer.styleMod import StyleMod

# Import itertools to slice the Config dictionary easily        
import itertools        

class NFFB(nn.Module):
    def __init__(self,config):
        super().__init__()
        GridEncoderConfig = dict(itertools.islice(config.items(), 13))
        self.xyz_encoder = FFB_encoder(GridEncoderConfig,config['freq_enc_type'],config['has_out'],config['bound'],config['layers_type'])
        self.feature_vector_size = self.xyz_encoder.embeddings_dim
        self.out_lin = nn.Linear(self.feature_vector_size,GridEncoderConfig['network_dims'][-1])
        self.styleTransferBlock = StyleMod(GridEncoderConfig['network_dims'][-1],self.feature_vector_size)
        #self.init_ouput([feature_vector_size]*config['n_levels'])
        
    def forward(self, x):
        """
            Inputs:
                x: [N, 3] - Input points 3D in [-scale,scale]
            Ouputs:
                out: (N), the final SDF estimation 
        """
        out_feat  = self.xyz_encoder(x)
        # Possible Modulation on last layer
        # out_feat_mod = self.styleTransferBlock(x,out_feat)
        out = self.out_lin(out_feat)  
        out = out_feat / (self.xyz_encoder.grid_levels+2)
        return out
    
    
        