import pytest
import torch

from sampling import top_k_generation, top_p_generation


@pytest.fixture
def logits() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 50)


def test_top_k_keeps_exactly_k(logits):
    _, masked = top_k_generation(logits, top_k=7, return_logits=True)
    finite = torch.isfinite(masked).sum(dim=-1)
    assert torch.all(finite == 7)


def test_top_k_respects_min_tokens(logits):
    _, masked = top_k_generation(logits, top_k=2, return_logits=True,
                                 min_tokens_to_keep=5)
    assert torch.all(torch.isfinite(masked).sum(dim=-1) == 5)


def test_top_k_batch_shape(logits):
    sample = top_k_generation(logits, top_k=10)
    assert sample.shape == (4,)
    assert sample.dtype == torch.long


def test_top_k_single_row_shape():
    torch.manual_seed(0)
    logits = torch.randn(50)
    sample = top_k_generation(logits, top_k=10)
    assert sample.shape == torch.Size([])


def test_top_p_keeps_required_mass(logits):
    _, masked = top_p_generation(logits, top_p=0.9, return_logits=True)
    probs = torch.softmax(masked, dim=-1)
    # every kept token has non-zero mass, dropped ones exactly zero
    kept = torch.isfinite(masked)
    assert torch.all(probs[~kept] == 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_top_p_minimal_set_property(logits):
    p = 0.9
    _, masked = top_p_generation(logits, top_p=p, return_logits=True)
    probs = torch.softmax(logits, dim=-1)
    for row in range(logits.shape[0]):
        kept = torch.isfinite(masked[row])
        kept_sorted = torch.sort(probs[row][kept], descending=True).values
        # kept set reaches p...
        assert float(kept_sorted.cumsum(0)[-1]) >= p - 1e-6
        # ...and dropping the smallest kept token would fall below p
        assert float(kept_sorted[:-1].cumsum(0)[-1]) < p + 1e-6


def test_top_p_invalid_value_raises(logits):
    with pytest.raises(ValueError, match="top_p"):
        top_p_generation(logits, top_p=0.0)


def test_temperature_flattens_distribution(logits):
    _, cold = top_k_generation(logits, top_k=50, return_logits=True,
                               temperature=0.5)
    _, hot = top_k_generation(logits, top_k=50, return_logits=True,
                              temperature=2.0)
    p_cold = torch.softmax(cold, dim=-1).max(dim=-1).values
    p_hot = torch.softmax(hot, dim=-1).max(dim=-1).values
    # higher temperature => flatter distribution => smaller peak probability
    assert torch.all(p_hot < p_cold)


def test_invalid_temperature_raises(logits):
    with pytest.raises(ValueError, match="temperature"):
        top_k_generation(logits, temperature=0.0)
