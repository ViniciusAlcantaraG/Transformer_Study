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
    def __init__(self, query_dim, key_dim, value_dim, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        self.scaled_dot_product = Attention(self.head_dim)
        # Compute Q, K and V at once for all heads
        self.query_layer = nn.Linear(query_dim, d_model)
        self.key_layer = nn.Linear(key_dim, d_model)
        self.value_layer = nn.Linear(value_dim, d_model)
        # Final Projection
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, query, key=None, value=None, mask=None):

        if key is None:
            key = query
        if value is None:
            value = key

        # Project x onto q, k, v
        batch_size, query_length, _ = query.shape
        key_length = key.shape[1]
        value_length = value.shape[1]
        if key_length != value_length:
            raise ValueError("Key and Value must have the same sequence length")
        
        q = self.query_layer(query)
        k = self.key_layer(key)
        v = self.value_layer(value)

        q = q.reshape(
            batch_size, query_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        k = k.reshape(
            batch_size, key_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        v = v.reshape(
            batch_size, value_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        values, attention = self.scaled_dot_product(q, k, v, mask)
        values = values.permute(0,2,1,3)
        values = values.reshape(batch_size, query_length, self.d_model)

        output = self.linear_layer(values)
        return output, attention

