"""
Benchmark the KV cache across context lengths.

Weights do not affect decode timing, so an untrained clone of the checkpoint
model is used with a longer positional encoding. This sweeps contexts well
past the trained 32-token window, where the cache's benefit becomes visible
instead of being drowned out by per-step Python/kernel overhead.

Examples:
    python benchmark_kv_cache.py                       # full sweep + plot
    python benchmark_kv_cache.py --n-tokens 32 64 128  # quick subset
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from generate import timed_decode_tokens
from transformer import Transformer


def default_repeats(n: int) -> int:
    """Long uncached decodes are slow; trade repetitions for sweep time."""
    if n <= 512:
        return 3
    return 2 if n <= 2048 else 1


def flops_per_decode(model_config: dict, n_new: int, use_cache: bool) -> float:
    """Predicted FLOPs for a decode of n_new tokens.

    Counts the linear layers (Q/K/V/out projections, FFN, output head) and
    the two attention matmuls; LayerNorm/residual/embedding ops are ignored,
    so both counts share the same small omission and the ratio is what
    matters. At step k the decoder processes k tokens uncached vs 1 token
    attending to k cached keys/values, which is exactly what the cache saves.
    """
    d = model_config["d_model"]
    f = model_config["dff"]
    layers = model_config["num_layers"]
    vocab = model_config["vocab_size"]
    token_sum = n_new * (n_new + 1) / 2                    # sum of k, k in 1..n
    square_sum = n_new * (n_new + 1) * (2 * n_new + 1) / 6  # sum of k^2

    if use_cache:
        # 1 query token per step: projections + FFN on it, attention over k keys
        per_layer = (8 * d * d + 4 * d * f) * n_new + 4 * d * token_sum
    else:
        # every step re-runs projections + FFN over all k tokens
        per_layer = (8 * d * d + 4 * d * f) * token_sum + 4 * d * square_sum
    head = 2 * d * vocab * n_new
    return layers * per_layer + head


def run_sweep(model: Transformer, model_config: dict, n_values: list[int],
              src_len: int, device: torch.device, repeats: int | None = None,
              log=print) -> list[dict]:
    """Time cached vs uncached decoding for each n, paired with FLOP theory."""
    vocab = model_config["vocab_size"]
    gen = torch.Generator().manual_seed(0)
    # ids 4..vocab-1 avoid the special tokens; the source is fixed across n
    src_ids = torch.randint(4, vocab, (1, src_len), generator=gen)

    rows: list[dict] = []
    for n in n_values:
        runs = repeats if repeats is not None else default_repeats(n)
        row: dict = {"n": n, "repeats": runs}
        seconds: dict[bool, float] = {}
        for use_cache in (False, True):
            timed_decode_tokens(model, src_ids, n, device, use_cache)  # warmup
            seconds[use_cache] = min(
                timed_decode_tokens(model, src_ids, n, device, use_cache)
                for _ in range(runs))
            key = "cached" if use_cache else "uncached"
            flops = flops_per_decode(model_config, n, use_cache)
            row[f"{key}_seconds"] = seconds[use_cache]
            row[f"{key}_flops"] = flops
            row[f"{key}_tok_s"] = n / seconds[use_cache]
            row[f"{key}_gflops_s"] = flops / seconds[use_cache] / 1e9
        row["speedup"] = seconds[False] / seconds[True]
        row["flop_ratio"] = row["uncached_flops"] / row["cached_flops"]
        rows.append(row)
        log(f"  n={n:5d} | uncached {row['uncached_tok_s']:9.1f} tok/s "
            f"({row['uncached_gflops_s']:6.2f} GFLOP/s) | "
            f"cached {row['cached_tok_s']:9.1f} tok/s "
            f"({row['cached_gflops_s']:6.2f} GFLOP/s) | "
            f"speedup {row['speedup']:6.2f}x (FLOP ratio {row['flop_ratio']:6.2f}x)")
    return rows


def plot_sweep(rows: list[dict], path: str | Path) -> None:
    """Save throughput and speedup curves against context length."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.loglog(ns, [r["uncached_tok_s"] for r in rows], marker="o",
               label="uncached")
    ax1.loglog(ns, [r["cached_tok_s"] for r in rows], marker="o",
               label="cached")
    ax1.set_xlabel("decoded tokens (n)")
    ax1.set_ylabel("tokens / s")
    ax1.set_title("Decode throughput vs context length")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.loglog(ns, [r["speedup"] for r in rows], marker="o",
               label="measured speedup")
    ax2.loglog(ns, [r["flop_ratio"] for r in rows], linestyle="--",
               label="predicted FLOP ratio")
    ax2.set_xlabel("decoded tokens (n)")
    ax2.set_ylabel("speedup (x)")
    ax2.set_title("KV-cache speedup vs context length")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--n-tokens", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--src-len", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=None,
                        help="timed runs per mode (default: 3 up to n=512, "
                             "2 up to 2048, 1 beyond)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"],
                        default="auto")
    parser.add_argument("--out", default="results/kv_cache_sweep.json")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = dict(ckpt["model_config"])
    n_values = sorted(set(args.n_tokens))
    if device.type == "cpu":
        # uncached decoding is cubic in n; keep a CPU sweep from running hours
        dropped = [n for n in n_values if n > 1024]
        if dropped:
            print(f"cpu device: skipping n > 1024 {dropped}")
            n_values = [n for n in n_values if n <= 1024]

    # untrained clone: weights don't affect decode speed, and a fresh model
    # can hold the long positional encodings the 32-token checkpoint cannot
    model_config["max_position_encod"] = max(n_values) + 1
    torch.manual_seed(0)
    model = Transformer(**model_config).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"KV-cache sweep | {device.type} | {n_params:,} params | "
          f"d_model={model_config['d_model']} "
          f"layers={model_config['num_layers']} | src_len={args.src_len} | "
          f"best of {'auto' if args.repeats is None else args.repeats}")
    with torch.no_grad():
        rows = run_sweep(model, model_config, n_values, args.src_len,
                         device, repeats=args.repeats)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "device": device.type,
        "model_config": model_config,
        "src_len": args.src_len,
        "results": rows,
    }, indent=2))
    print(f"wrote {out}")
    if not args.no_plot:
        plot_path = out.with_suffix(".png")
        plot_sweep(rows, plot_path)
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
