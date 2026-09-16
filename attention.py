import torch
from torch import nn
import math

class Attention(nn.Module):

    def __init__(self, head_dim):
        super().__init__()
        self.scale = 1.0/math.sqrt(head_dim)

    def forward(self, query, key, values, mask):

        scores = torch.matmul(query, key.transpose(-2,-1))
        scores = scores * self.scale
        if mask is not None:
            scores = scores.masked_fill_(mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        scores = torch.matmul(attention, values)
        return scores, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        self.scaled_dot_product = Attention(self.head_dim)
        # Compute Q, K and V at once for all heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        # Final Projection
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        batch_size, sequence_length, input_dim = x.size()

        # Project x onto q, k, v
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3*self.head_dim)
        # Rearrange to (batch size, num_heads, seq_length, 3*head_dim)
        qkv = qkv.permute(0,2,1,3)

        # Split the last dimension into q, k, v
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = self.scaled_dot_product(q, k, v, mask)
        values = values.permute(0,2,1,3)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

        output = self.linear_layer(values)
        return output

