import pytest
import torch

from transformer import Transformer, positional_encoding


def test_forward_shape(tiny_model, batches):
    src, tgt = batches
    out = tiny_model(src, tgt)
    assert out.shape == (2, 4, tiny_model.vocab_size)


def test_sequence_longer_than_positions_raises(tiny_model):
    src = torch.full((1, 20), 2)
    tgt = torch.full((1, 20), 2)
    with pytest.raises(ValueError, match="positional encoding"):
        tiny_model(src, tgt)


def test_positional_encoding_is_a_registered_buffer(tiny_model):
    buffers = dict(tiny_model.named_buffers())
    assert "encoder.pos_encoding" in buffers
    assert "decoder.pos_encoding" in buffers


def test_positional_encoding_values():
    pe = positional_encoding(8, 4)
    assert pe.shape == (4, 8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4))
    assert torch.allclose(pe[0, 1::2], torch.ones(4))
    # bounded in [-1, 1]
    assert pe.abs().max() <= 1.0


def test_gradients_reach_embeddings(tiny_model, batches):
    src, tgt = batches
    out = tiny_model(src, tgt)
    out.sum().backward()
    assert tiny_model.encoder.embedding.weight.grad is not None
    assert tiny_model.decoder.embedding.weight.grad is not None
    assert tiny_model.linear.weight.grad is not None


def test_decoder_start_pos_out_of_range_raises(tiny_model, batches):
    src, _ = batches
    enc = tiny_model.encoder(src)
    with pytest.raises(ValueError, match="positional encoding"):
        tiny_model.decoder(torch.tensor([[2]]), enc, start_pos=16)


def test_attention_weights_available_from_decoder(tiny_model, batches):
    src, tgt = batches
    enc = tiny_model.encoder(src)
    _, weights = tiny_model.decoder(tgt, enc)
    assert "layer1_self" in weights and "layer1_cross" in weights
    assert weights["layer1_cross"].shape == (2, 2, 4, 4)
