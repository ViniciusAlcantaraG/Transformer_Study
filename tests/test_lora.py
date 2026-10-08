import torch
from torch import nn

from lora import LinearWithLoRA, LoRA_Layer


def test_adapter_output_is_zero_at_init():
    lora = LoRA_Layer(input_dim=8, output_dim=8, rank=2, alpha=4.0)
    # B is initialized to zero, so the adapter contributes nothing initially
    out = lora(torch.randn(3, 8))
    assert torch.allclose(out, torch.zeros(3, 8))


def test_output_matches_base_linear_at_init():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    wrapped = LinearWithLoRA(base, rank=2, alpha=4.0)
    x = torch.randn(5, 8)
    assert torch.allclose(wrapped(x), base(x), atol=1e-6)


def test_base_weights_are_frozen():
    base = nn.Linear(8, 8)
    wrapped = LinearWithLoRA(base, rank=2, alpha=4.0)
    assert all(not p.requires_grad for p in base.parameters())
    assert all(p.requires_grad for p in wrapped.lora.parameters())


def test_gradients_flow_only_to_adapter():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    wrapped = LinearWithLoRA(base, rank=2, alpha=4.0)
    x = torch.randn(4, 8)
    wrapped(x).sum().backward()
    assert base.weight.grad is None
    assert wrapped.lora.A.grad is not None
    assert wrapped.lora.B.grad is not None


def test_adapter_can_learn_nonzero_contribution():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    wrapped = LinearWithLoRA(base, rank=2, alpha=4.0)
    x = torch.randn(16, 8)
    target = torch.randn(16, 8)
    opt = torch.optim.Adam(wrapped.lora.parameters(), lr=0.05)
    for _ in range(100):
        opt.zero_grad()
        loss = ((wrapped(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        base_loss = ((base(x) - target) ** 2).mean()
    assert float(loss) < float(base_loss)
