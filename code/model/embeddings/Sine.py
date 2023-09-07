import torch
from torch import nn

import numpy as np
class Sine(nn.Module):
    def __init__(self, w0):
        super(Sine,self).__init__()
        self.w0 = w0
        for param in self.parameters():
            param.requires_grad = True
    def forward(self, input):
        return torch.sin(input * self.w0)

def sine_init(m, w0, num_input=None):
    if hasattr(m, 'weight'):
        if num_input is None:
            num_input = m.weight.size(-1)
            torch.nn.init.uniform_(m.weight, -np.sqrt(6 / num_input)/w0 ,np.sqrt(6 / num_input)/w0)
            torch.nn.init.uniform_(m.bias, -np.sqrt(6 / num_input)/w0 ,np.sqrt(6 / num_input)/w0)
        
def first_layer_sine_init(m):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        torch.nn.init.uniform_(m.weight,-1.0 / num_input, 1.0 / num_input)
        torch.nn.init.uniform_(m.bias,-1.0 / num_input, 1.0 / num_input)





