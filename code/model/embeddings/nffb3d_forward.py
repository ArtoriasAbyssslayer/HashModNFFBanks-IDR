# 3D shape fitting encoder based on nffb 

import torch 
import torch.nn as nn

from model.embeddings.nffb3d import FourierFilterBanks as FFB_encoder
from model.embeddings.tcnn_src.Sine import sine_init
from model.embeddings.stylemod import styleMod
        
        

class NFFB(nn.Module):
    def __init__(self,config):
        self.xyz_encoder = FFB_encoder(GridEncoderConfig,HashEncoderType,d_in, boudn, has_out)
        self.feature_vector_size = self.xyz_encoder.embedding_dim
        self.out_lin = nn.Linear(enc_out_dim, 1)
        styleTransferBlock = styleMod(feature_vector_size)
        self.init_ouput([feature_vector_size]*config['n_levels'])
        
    @torch.no_grad    
    def forward(self, x):
        """
            Inputs:
                x: [N, 3] - Input points 3D in [-scale,scale]
            Ouputs:
                out: (N), the final SDF estimation 
        """
        out  = self.xyz_encoder(x)
        out_feat = torch.cat(out,dim=-1)
        # Possible Modulation on last layer
        out_feat_mod = self.styleTransferBlock(x,out_feat)
        out_feat = self.out_lin(out_feat_mod)  
        out = out_feat / self.xyz_encoder.grid_level
        return out
    
    
        