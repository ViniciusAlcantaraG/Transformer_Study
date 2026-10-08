import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from tokenizer import BPETokenizer
from transformer import Transformer


@pytest.fixture
def tiny_model() -> Transformer:
    torch.manual_seed(0)
    model = Transformer(
        d_model=32,
        dff=64,
        num_layers=1,
        num_heads=2,
        vocab_size=512,  # byte-level BPE vocab starts at 4 + 256 symbols
        max_position_encod=16,
        p_rate=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def batches() -> tuple[torch.Tensor, torch.Tensor]:
    # id 0 = <pad>, 2 = <bos>, other ids are ordinary tokens
    src = torch.tensor([[2, 5, 6, 0], [2, 7, 0, 0]])
    tgt = torch.tensor([[2, 8, 9, 0], [2, 8, 0, 0]])
    return src, tgt


@pytest.fixture
def unpadded_batches() -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.tensor([[2, 5, 6, 7], [2, 8, 9, 10]])
    tgt = torch.tensor([[2, 11, 12, 13], [2, 14, 15, 16]])
    return src, tgt


@pytest.fixture
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(["hello world", "ola mundo", "the cat sat"],
                              num_merges=60)
