import torch
import torch.nn as nn

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x: input tensor with shape (sequence_length, batch_size, embedding_size)
        batch_size = x.size(1)
        x_reshaped = x.permute(1, 0, 2)  # Shape: (batch_size, sequence_length, embedding_size)
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output