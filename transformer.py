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
        self.attention = MultiHeadAttention(d_model, d_model, d_model, d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.dropout1 = nn.Dropout(self.p_rate)
        self.dropout2 = nn.Dropout(self.p_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        attention_results, attn = self.attention(x, mask=mask)
        attention_results = self.dropout1(attention_results)
        out_1 = self.layernorm1(x + attention_results)
        ffn_results = self.ffn(out_1)
        ffn_results = self.dropout2(ffn_results)
        out_2 = self.layernorm2(out_1 + ffn_results)
        return out_2, attn

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
        for index, layer in enumerate(self.enc_layers):
            x,_ = layer(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, dff, num_heads, p_rate):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model, d_model, d_model, d_model, num_heads)
        self.cross_attention = MultiHeadAttention(
            d_model, d_model, d_model, d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)

        self.dropout1 = nn.Dropout(p_rate)
        self.dropout2 = nn.Dropout(p_rate)
        self.dropout3 = nn.Dropout(p_rate)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None, use_cache=True):

        self_output, self_weights = self.self_attention(x, mask=look_ahead_mask, use_cache=use_cache)
        x = self.layernorm1(x+self.dropout1(self_output))
        cross_output, cross_weights = self.cross_attention(query=x,key=encoder_output,value=encoder_output,mask=padding_mask, use_cache=use_cache)
        x = self.layernorm2(x+self.dropout2(cross_output))

        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout3(ffn_output))

        return x, self_weights, cross_weights
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, dff, num_heads, vocab_size, max_position_encod, p_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_position_encod = max_position_encod
        self.p_rate = p_rate
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p_rate)
        self.dec_layers = nn.ModuleList([DecoderBlock(d_model, dff, num_heads, p_rate) for _ in range(num_layers)])
        self.register_buffer("pos_encoding", positional_encoding(d_model, max_position_encod))

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None, use_cache=True):

        seq_len = x.shape[1]
        attention_weights = {}
        if seq_len > self.pos_encoding.shape[0]:
            raise ValueError("Input sequence exceeds maximum positional encoding length")
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x += self.pos_encoding[:seq_len]
        x = self.dropout(x)
        for index, layer in enumerate(self.dec_layers):
            x, self_weights, cross_weights = layer(x, encoder_output, look_ahead_mask, padding_mask, use_cache)
            attention_weights[f"layer{index+1}_self"]=self_weights
            attention_weights[f"layer{index+1}_cross"]=cross_weights
        return x, attention_weights

class Transformer(nn.Module):
    def __init__(self, d_model, dff, num_layers, num_heads, vocab_size, max_position_encod, p_rate):
        super().__init__()

        self.d_model = d_model
        self.dff = dff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_position_encod = max_position_encod
        self.p_rate = p_rate
        self.encoder = Encoder(num_layers, d_model, dff, num_heads, vocab_size, max_position_encod, p_rate)
        self.decoder = Decoder(num_layers, d_model, dff, num_heads, vocab_size, max_position_encod, p_rate)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, source, target, mask=None, look_ahead_mask=None, padding_mask=None, use_cache=True):

        encoder_output = self.encoder(source, mask=mask)
        decoder_output, attention_weights = self.decoder(target, encoder_output, look_ahead_mask, padding_mask, use_cache)
        output = self.linear(decoder_output)
        output = torch.softmax(output, dim=-1)
        return output