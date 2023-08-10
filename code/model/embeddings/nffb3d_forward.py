# 3D shape fitting encoder based on nffb 

import torch 
import torch.nn as nn

from model.embeddings.nffb import FourierFilterBanks as FFB_encoder
from model.embeddings.tcunn_implementations.Sine import sine_init
        
        

class NFFB(nn.Module):
    def __init__(self,config):
        self.xyz_encoder = FFB_encoder(GridEncoderConfig,HashEncoderType,d_in, boudn, has_out)
        self.ebmedding_dim = self.xyz_encoder.embedding_dim
        self.out_lin = nn.Linear(enc_out_dim, 1)

        self.init_ouput([256, 256, 256, 256, 256, 256])
        
        
        
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
        out_feat = self.out_lin(out_feat)
        out = out_feat / self.xyz_encoder.grid_level
        
        return out
    
    
        @torch.no_grad
        
        # optimizer utils
        def get_optimizer(self,lr,weight_decay):
            return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)  