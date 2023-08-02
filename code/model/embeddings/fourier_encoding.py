import torch
import torch.nn as nn
class FourierEncoding(nn.Module):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        self.include_input = kwargs['include_input']
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
        self.embeddings_dim = out_dim
    # Custom embed function in order to use to for simple Frequency Embedding
    def embed(self,inputs):
        ff_embeds= torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        if self.include_input:
            return torch.cat([inputs, ff_embeds], -1)
        else:
            return ff_embeds
    def __call__(self,inputs):
        return self.embed(inputs=inputs)
        
    
def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = FourierEncoding(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

