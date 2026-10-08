"""
Examples:
    python generate.py --source "hello world" --checkpoint checkpoints/best.pt
    python generate.py --source "hello" --strategy topk --top-k 20 --temperature 0.8
    python generate.py --benchmark --checkpoint checkpoints/best.pt
"""

import argparse
import sys
import time

import torch
from torch import Tensor

from sampling import top_k_generation, top_p_generation
from tokenizer import BPETokenizer
from training import causal_mask, make_masks
from transformer import Transformer


def load_checkpoint(path: str, device: torch.device) -> tuple[Transformer, BPETokenizer]:
    """Rebuild model + tokenizer from a checkpoint saved by training.py."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Transformer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = BPETokenizer.from_state(ckpt["tokenizer"])
    return model, tokenizer


def _sample_next(logits: Tensor, strategy: str, temperature: float,
                 top_k: int, top_p: float) -> Tensor:
    if strategy == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)
    if strategy == "topk":
        tok = top_k_generation(logits, top_k=top_k, temperature=temperature)
    elif strategy == "topp":
        tok = top_p_generation(logits, top_p=top_p, temperature=temperature)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return tok.reshape(logits.shape[:-1] + (1,))


@torch.no_grad()
def generate(model: Transformer, tokenizer: BPETokenizer, source_text: str,
             device: torch.device, strategy: str = "greedy",
             max_new_tokens: int = 16, temperature: float = 1.0,
             top_k: int = 50, top_p: float = 0.9,
             use_cache: bool = True, stop_on_eos: bool = True) -> str:
    """Translate `source_text`, decoding one token at a time.

    With use_cache=True the encoder runs once and each decoding step only
    computes attention for the newest token (positions come from start_pos).
    """
    max_len = model.decoder.max_position_encod
    # bos occupies position 0, so at most max_len - 1 new tokens fit
    max_new_tokens = min(max_new_tokens, max_len - 1)
    src = torch.tensor([tokenizer.encode(source_text, max_len)],
                       dtype=torch.long, device=device)
    encoder_mask, _, cross_mask = make_masks(src, torch.zeros_like(src))

    model.clear_cache()
    encoder_output = model.encoder(src, mask=encoder_mask)

    ys = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    tokens = []
    for _ in range(max_new_tokens):
        if use_cache:
            out, _ = model.decoder(
                ys[:, -1:], encoder_output,
                look_ahead_mask=None, padding_mask=cross_mask,
                use_cache=True, start_pos=ys.shape[1] - 1,
            )
        else:
            look = causal_mask(ys.shape[1]).unsqueeze(0).unsqueeze(0).to(ys.device)
            out, _ = model.decoder(
                ys, encoder_output,
                look_ahead_mask=look, padding_mask=cross_mask, use_cache=False,
            )
        logits = model.linear(out[:, -1])
        next_id = _sample_next(logits, strategy, temperature, top_k, top_p)
        ys = torch.cat([ys, next_id], dim=1)
        if stop_on_eos and int(next_id.item()) == tokenizer.eos_id:
            break
        tokens.append(int(next_id.item()))
    model.clear_cache()
    return tokenizer.decode(tokens)


@torch.no_grad()
def _timed_decode(model: Transformer, tokenizer: BPETokenizer,
                  source_text: str, device: torch.device, use_cache: bool,
                  n_tokens: int) -> float:
    """Generate a fixed number of tokens (ignoring EOS) and return seconds."""
    max_len = model.decoder.max_position_encod
    src = torch.tensor([tokenizer.encode(source_text, max_len)],
                       dtype=torch.long, device=device)
    encoder_mask, _, cross_mask = make_masks(src, torch.zeros_like(src))

    model.clear_cache()
    encoder_output = model.encoder(src, mask=encoder_mask)
    ys = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_tokens):
        if use_cache:
            out, _ = model.decoder(
                ys[:, -1:], encoder_output,
                look_ahead_mask=None, padding_mask=cross_mask,
                use_cache=True, start_pos=ys.shape[1] - 1,
            )
        else:
            look = causal_mask(ys.shape[1]).unsqueeze(0).unsqueeze(0).to(ys.device)
            out, _ = model.decoder(
                ys, encoder_output,
                look_ahead_mask=look, padding_mask=cross_mask, use_cache=False,
            )
        logits = model.linear(out[:, -1])
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_id], dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.clear_cache()
    return elapsed


def benchmark(model: Transformer, tokenizer: BPETokenizer, source_text: str,
              device: torch.device, n_tokens: int = 64,
              repeats: int = 3) -> dict[bool, float]:
    n_tokens = min(n_tokens, model.decoder.max_position_encod - 1)
    results: dict[bool, float] = {}
    for use_cache in (False, True):
        times = [_timed_decode(model, tokenizer, source_text, device,
                               use_cache, n_tokens) for _ in range(repeats)]
        best = min(times)
        results[use_cache] = n_tokens / best
    speedup = results[True] / results[False]

    print(f"\nBenchmark: {n_tokens} tokens, best of {repeats}, "
          f"device={device.type}")
    print(f"  uncached : {results[False]:8.1f} tokens/s")
    print(f"  cached   : {results[True]:8.1f} tokens/s")
    print(f"  speedup  : {speedup:8.2f}x")
    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--source", default="hello world")
    parser.add_argument("--strategy", choices=["greedy", "topk", "topp"],
                        default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--no-cache", action="store_true",
                        help="decode without the KV cache")
    parser.add_argument("--benchmark", action="store_true",
                        help="compare cached vs uncached decoding speed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    if args.benchmark:
        benchmark(model, tokenizer, args.source, device)
        return

    text = generate(
        model, tokenizer, args.source, device,
        strategy=args.strategy, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        use_cache=not args.no_cache,
    )
    print(f"source : {args.source}")
    print(f"output : {text}")


if __name__ == "__main__":
    main()
