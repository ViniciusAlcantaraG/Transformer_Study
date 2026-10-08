import pytest
import torch

from attention import MultiHeadAttention


def test_output_shape():
    mha = MultiHeadAttention(32, 32, 32, 32, num_heads=4)
    x = torch.randn(2, 6, 32)
    out, weights = mha(x)
    assert out.shape == (2, 6, 32)
    assert weights.shape == (2, 4, 6, 6)


def test_heads_partition_dimension():
    # 32 dims over 4 heads => 8 dims per head, weights sum to 1 per head
    mha = MultiHeadAttention(32, 32, 32, 32, num_heads=4)
    x = torch.randn(2, 5, 32)
    _, weights = mha(x)
    assert mha.head_dim == 8
    totals = weights.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)


def test_indivisible_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(30, 30, 30, 30, num_heads=4)


def test_masked_keys_get_zero_attention():
    mha = MultiHeadAttention(16, 16, 16, 16, num_heads=2)
    x = torch.randn(1, 4, 16)
    # only key position 0 is visible to every query
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    mask[..., 0] = False
    _, weights = mha(x, mask=mask)
    assert torch.allclose(weights[..., 0], torch.ones(1, 2, 4), atol=1e-6)
    assert torch.allclose(weights[..., 1:], torch.zeros(1, 2, 4, 3), atol=1e-6)


def test_gradients_reach_projections():
    mha = MultiHeadAttention(16, 16, 16, 16, num_heads=2)
    x = torch.randn(2, 3, 16, requires_grad=True)
    out, _ = mha(x)
    out.sum().backward()
    assert x.grad is not None
    assert mha.query_layer.weight.grad is not None
    assert mha.linear_layer.weight.grad is not None
