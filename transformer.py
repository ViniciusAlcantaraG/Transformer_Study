import math

import torch
from torch import Tensor, nn
from attention import MultiHeadAttention
from ffn import FeedForward


def positional_encoding(d_model: int, max_position: int) -> Tensor:
    """Word embeddings don't account for position, so we use sinusoidal waves
    to encode it. Lower dimensions change values quickly with frequence while
    higer dimensions are more stable. This guarantees unique representation for
    each position"""
    positions = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000)/d_model))

    pe = torch.zeros(max_position, d_model)
    pe[:,0::2] = torch.sin(positions*div_term)
    pe[:,1::2] = torch.cos(positions*div_term[:pe[:, 1::2].shape[1]])
    return pe


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, num_heads: int, p_rate: float) -> None:
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

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:

        attention_results, attn = self.attention(x, mask=mask)
        attention_results = self.dropout1(attention_results)
        out_1 = self.layernorm1(x + attention_results)
        ffn_results = self.ffn(out_1)
        ffn_results = self.dropout2(ffn_results)
        out_2 = self.layernorm2(out_1 + ffn_results)
        return out_2, attn

class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, dff: int, num_heads: int,
                 vocab_size: int, max_position_encod: int, p_rate: float = 0.1) -> None:
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

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:

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
    def __init__(self, d_model: int, dff: int, num_heads: int, p_rate: float) -> None:
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

    def forward(self, x: Tensor, encoder_output: Tensor,
                look_ahead_mask: Tensor | None = None,
                padding_mask: Tensor | None = None,
                use_cache: bool = False) -> tuple[Tensor, Tensor, Tensor]:

        self_output, self_weights = self.self_attention(x, mask=look_ahead_mask, use_cache=use_cache)
        x = self.layernorm1(x+self.dropout1(self_output))
        # Cross-attention keys/values come from the encoder output, which is identical
        # on every decoding step, so they are never cached (caching would duplicate them).
        cross_output, cross_weights = self.cross_attention(query=x,key=encoder_output,value=encoder_output,mask=padding_mask)
        x = self.layernorm2(x+self.dropout2(cross_output))

        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout3(ffn_output))

        return x, self_weights, cross_weights
class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, dff: int, num_heads: int,
                 vocab_size: int, max_position_encod: int, p_rate: float = 0.1) -> None:
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

    def forward(self, x: Tensor, encoder_output: Tensor,
                look_ahead_mask: Tensor | None = None,
                padding_mask: Tensor | None = None,
                use_cache: bool = False,
                start_pos: int = 0) -> tuple[Tensor, dict[str, Tensor]]:

        seq_len = x.shape[1]
        attention_weights: dict[str, Tensor] = {}
        if start_pos + seq_len > self.pos_encoding.shape[0]:
            raise ValueError("Input sequence exceeds maximum positional encoding length")
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        # start_pos lets incremental decoding (KV cache) give each new token its
        # true position instead of re-using pe[:seq_len] starting at 0 every step
        x += self.pos_encoding[start_pos : start_pos + seq_len]
        x = self.dropout(x)
        for index, layer in enumerate(self.dec_layers):
            x, self_weights, cross_weights = layer(x, encoder_output, look_ahead_mask, padding_mask, use_cache)
            attention_weights[f"layer{index+1}_self"]=self_weights
            attention_weights[f"layer{index+1}_cross"]=cross_weights
        return x, attention_weights

class Transformer(nn.Module):
    def __init__(self, d_model: int, dff: int, num_layers: int, num_heads: int,
                 vocab_size: int, max_position_encod: int, p_rate: float) -> None:
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

    def forward(self, source: Tensor, target: Tensor, mask: Tensor | None = None,
                look_ahead_mask: Tensor | None = None,
                padding_mask: Tensor | None = None,
                use_cache: bool = False) -> Tensor:

        encoder_output = self.encoder(source, mask=mask)
        decoder_output, attention_weights = self.decoder(target, encoder_output, look_ahead_mask, padding_mask, use_cache)
        output = self.linear(decoder_output)
        return output

    def clear_cache(self) -> None:
        """Reset the decoder's KV cache. Must be called between independently
        generated sequences when decoding with use_cache=True."""
        for layer in self.decoder.dec_layers:
            layer.self_attention.kv_cache.clear_cache()
