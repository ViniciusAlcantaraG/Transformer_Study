import json
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from training import (build_model_config, collate_batch, evaluate,
                      plot_history, shift_targets)


def test_build_model_config_roundtrip():
    cfg = build_model_config(vocab_size=50, max_len=16)
    assert cfg["vocab_size"] == 50
    assert cfg["max_position_encod"] == 16
    assert set(cfg) == {"d_model", "dff", "num_layers", "num_heads",
                        "vocab_size", "max_position_encod", "p_rate"}


def test_shift_targets_is_next_token_prediction():
    tgt = torch.tensor([[2, 5, 6, 3, 0], [2, 7, 3, 0, 0]])
    dec_in, labels = shift_targets(tgt)
    assert torch.equal(dec_in, tgt[:, :-1])
    assert torch.equal(labels, tgt[:, 1:])
    # labels are what the decoder input should predict one step ahead
    assert torch.all(labels[:, :-1] == dec_in[:, 1:])
    # <eos> is the final non-pad label of each row
    assert int(labels[0, 2]) == 3
    assert int(labels[1, 1]) == 3
    # pad positions stay pads so ignore_index still skips them
    assert int(labels[0, 3]) == 0
    assert int(labels[1, 2]) == 0


def test_evaluate_returns_loss_and_perplexity(tiny_model, tokenizer, batches):
    src, tgt = batches
    pairs = [(s.clone(), t.clone()) for s, t in zip(src, tgt)]
    loader = DataLoader(pairs, batch_size=2, collate_fn=collate_batch)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    loss, ppl = evaluate(tiny_model, loader, criterion, torch.device("cpu"))
    assert math.isfinite(loss) and math.isfinite(ppl)
    assert abs(math.exp(loss) - ppl) < 1e-6


def test_evaluate_does_not_track_gradients(tiny_model, tokenizer, batches):
    src, tgt = batches
    pairs = [(s.clone(), t.clone()) for s, t in zip(src, tgt)]
    loader = DataLoader(pairs, batch_size=2, collate_fn=collate_batch)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    tiny_model.zero_grad()
    evaluate(tiny_model, loader, criterion, torch.device("cpu"))
    assert all(p.grad is None for p in tiny_model.parameters())


def test_plot_history_writes_png(tmp_path):
    history = {"train": [3.0, 2.0], "val": [3.2, 2.4], "val_ppl": [24.5, 11.0]}
    path = tmp_path / "curve.png"
    plot_history(history, path)
    assert path.exists() and path.stat().st_size > 0


def test_history_json_serializable(tmp_path):
    history = {"train": [1.0], "val": [1.1], "val_ppl": [3.0]}
    p = tmp_path / "h.json"
    p.write_text(json.dumps(history))
    assert json.loads(p.read_text())["val"] == [1.1]
