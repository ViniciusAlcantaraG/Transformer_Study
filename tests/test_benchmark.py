import json
import math

import torch

from benchmark_kv_cache import flops_per_decode, run_sweep
from generate import timed_decode_tokens


def _config(**overrides) -> dict:
    cfg = {"d_model": 32, "dff": 64, "num_layers": 1, "num_heads": 2,
           "vocab_size": 512, "max_position_encod": 16, "p_rate": 0.0}
    cfg.update(overrides)
    return cfg


def test_flops_uncached_exceeds_cached():
    cfg = _config()
    for n in (4, 16, 64):
        uncached = flops_per_decode(cfg, n, use_cache=False)
        cached = flops_per_decode(cfg, n, use_cache=True)
        assert uncached > cached > 0


def test_flop_ratio_grows_with_n():
    # the cache saves O(n^2) attention + O(n^2) FFN recomputation per decode,
    # so the predicted advantage must widen as contexts lengthen
    cfg = _config()
    short = flops_per_decode(cfg, 16, False) / flops_per_decode(cfg, 16, True)
    long = flops_per_decode(cfg, 256, False) / flops_per_decode(cfg, 256, True)
    assert long > short


def test_flops_scale_with_output_head():
    # vocab-projection cost is per generated token in both modes
    cfg, big = _config(), _config(vocab_size=1024)
    assert (flops_per_decode(big, 32, True) - flops_per_decode(cfg, 32, True)
            == 2 * 32 * (1024 - 512) * 32)


def test_timed_decode_tokens_runs_both_modes(tiny_model):
    src = torch.tensor([[5, 6, 7, 8]])
    for use_cache in (False, True):
        seconds = timed_decode_tokens(tiny_model, src, 4,
                                      torch.device("cpu"), use_cache)
        assert seconds > 0


def test_run_sweep_smoke(tiny_model):
    rows = run_sweep(tiny_model, _config(), n_values=[4, 8], src_len=4,
                     device=torch.device("cpu"), repeats=1,
                     log=lambda *_: None)
    assert [r["n"] for r in rows] == [4, 8]
    for row in rows:
        assert row["uncached_tok_s"] > 0 and row["cached_tok_s"] > 0
        assert math.isfinite(row["speedup"]) and row["speedup"] > 0
        assert row["uncached_flops"] > row["cached_flops"]
    json.dumps(rows)  # results must serialize to the sweep's JSON report
