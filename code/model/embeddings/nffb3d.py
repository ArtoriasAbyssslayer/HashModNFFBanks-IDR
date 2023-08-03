# 3D shape fitting encoder based on nffb 

import torch 
import torch.nn as nn

from model.embeddings.nffb import FourierFilterBanks as FFB_encoder
from model.embeddings.tcunn_implementations.Sine import sine_init


        
        
        
        

class NFFB(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.3d_encoder = FFB_encoder(GridEncoderNetConfig=config,bound=0.5)

        enc_out_dim = self.3d_encoder.output_dim
        
        self.out_lin = nn.Linear(enc_out_dim, 1)

        self.init_ouput([256, 256, 256, 256, 256, 256])
        
        
        
        
    def forward(self, x):
        """
            Inputs:
                x: [N, 3] - Input points 3D in [-scale,scale]
            Ouputs:
                out: (N), the final SDF estimation 
        """
        out  = self.3d_encoder(x)
        out_feat = torch.cat(out,dim=-1)
        out_feat = self.out_lin(out_feat)
        out = out_feat / self.xyz_encoder.grid_level
        
        return out
    
    
        @torch.no_grad
        
        # optimizer utils
        
        def get_params(selg