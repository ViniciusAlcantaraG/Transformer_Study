import torch
from torch import nn
from attention import MultiHeadAttention
from ffn import FeedForward
import math


def positional_encoding(d_model, max_position):

    positions = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000)/d_model))

    pe = torch.zeros(max_position, d_model)
    pe[:,0::2] = torch.sin(positions*div_term)
    pe[:,1::2] = torch.cos(positions*div_term[:pe[:, 1::2].shape[1]])
    return pe


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dff, num_heads, p_rate):
        super().__init__()

        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.p_rate = p_rate
        self.attention = MultiHeadAttention(d_model, d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.dropout1 = nn.Dropout(self.p_rate)
        self.dropout2 = nn.Dropout(self.p_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        attention_results = self.attention(x, mask)
        attention_results = self.dropout1(attention_results)
        out_1 = self.layernorm1(x + attention_results)
        ffn_results = self.ffn(out_1)
        ffn_results = self.dropout2(ffn_results)
        out_2 = self.layernorm2(out_1 + ffn_results)
        return out_2

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, dff, num_heads, vocab_size, max_position_encod, p_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_position_encod = max_position_encod
        self.p_rate = p_rate
        self.pos_encoding = positional_encoding(d_model, max_position_encod)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p_rate)
        self.enc_layers = nn.ModuleList([TransformerBlock(d_model, dff, num_heads, p_rate) for _ in range(num_layers)])
        self.register_buffer("pos_encoding", positional_encoding(d_model, max_position_encod))

    def forward(self, x, mask=None):

        seq_len = x.shape[1]
        if seq_len > self.pos_encoding.shape[0]:
            raise ValueError("Input sequence exceeds maximum positional encoding length")
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x += self.pos_encoding[:seq_len]
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x






