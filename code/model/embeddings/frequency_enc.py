import torch
import torch.nn as nn
import numpy as np
class NerfPositionalEncoding(nn.Module):
    def __init__(self, include_input:bool, dim: int, n_levels: int = 10):
        """
            Positional encoding from NeRF
            [sin(x),cos(x), sin(4x), cos(4x), sin(8x), cos(8x)
             .....,sin(2^n*x), cos(2^n*x)]

            Args:
                - dim(int) : input dimensions
                - n_levels(int,optional): frequency bands number
        """
        super().__init__()
        self.n_levels = n_levels
        assert self.n_levels > 0
        self.include_input = include_input
        freqs = 2. ** torch.linspace(0., n_levels - 1, n_levels)
        self.register_buffer('freqs', freqs)

        # ---
        self.input_dim = dim
        self.output_dim = dim * n_levels * 2
        self.embeddings_dim = self.output_dim  + self.input_dim if self.include_input else self.output_dim
    def forward(self, x: torch.Tensor):
        # unsqueeze in order to multiply it with freqs  space
        input = x 
        x = x.unsqueeze(dim=-1)  # (...,dim,1)
        x = x * self.freqs.to(input.device)  # (..., dim,L)
        # concatenate the sin,cos input frequencies series
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1)  # (..,dim,L*2)
        if self.include_input:
            return torch.cat((input,x.flatten(-2, -1)),dim=-1)  # (...,dim*L*2) - flatten the tensor to 1D
        else:
            return x.flatten(-2, -1)

class PositionalEncoding(nn.Module):
    ''' IDR classic method for positional encoding '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.include_input = kwargs['include_input']
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
        out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    # Custom embed function in order to use to for simple Frequency Embedding
    def embed(self,inputs):
        ff_embeds= torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        if self.include_input:
            return torch.cat([inputs, ff_embeds], -1)
        else:
            return ff_embeds
    def __call__(self,inputs):
        return self.embed(inputs=inputs)
        
#Simple Fourier Feature Encoder 
class FourierFeature(nn.Module):
    '''Fourrier Feature Encoder'''
    def __init__(self, channels, sigma=1.0, input_dims=3, include_input=True) -> None:
        super().__init__()
        self.register_buffer('B', torch.randn(input_dims, channels) * sigma, True)
        self.channels = channels
        self.embeddings_dim  = 2 * self.channels + 3 if include_input else 2 * self.channels
        self.include_input = include_input
    def forward(self, x):
        xp = torch.matmul(2 * np.pi * x, self.B.to(x.device))
        return torch.cat([x, torch.sin(xp), torch.cos(xp)], dim=-1) if self.include_input else torch.cat([torch.sin(xp), torch.cos(xp)], dim=-1)
#SHencoder 
class SHEncoder(nn.Module):
    '''Spherical Harmonics Encoder'''
    def __init__(self, input_dims=3, degree=4):
        
        super().__init__()

        self.input_dims = input_dims
        self.degree = degree

        assert self.input_dims == 3
        assert self.degree >= 1 and self.degree <= 5
        # embeddings_Dim == calculated dim of the output
        self.embeddings_dim  = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.embeddings_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                #result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
    

   
def get_embedder(multires):
    '''Default Get embedder function for the Positional Encoding -> Used original IDR configs'''
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires+3,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = PositionalEncoding(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

