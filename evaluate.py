"""
Examples:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --limit 50
"""

import argparse
import json
import sys
from pathlib import Path

import sacrebleu
import torch

from generate import generate, load_checkpoint
from training import MAX_LEN, N_VAL, load_opus_splits


def corpus_bleu(hypotheses: list[str], references: list[str]) -> tuple[float, str]:
    """Corpus BLEU-4 on detokenized text.

    sacrebleu applies its own fixed 13a tokenization to the raw strings, so
    the score depends only on the decoded text, never on the model's tokenizer.
    """
    metric = sacrebleu.metrics.BLEU()
    result = metric.corpus_score(hypotheses, [references])
    return result.score, str(metric.get_signature())


@torch.no_grad()
def translate_sources(model, tokenizer, sources: list[str], device: torch.device,
                      max_new_tokens: int = MAX_LEN - 1, strategy: str = "greedy",
                      limit: int | None = None) -> list[str]:
    """Greedy/sample a translation for each source string (stops at EOS)."""
    if limit is not None:
        sources = sources[:limit]
    hypotheses = []
    for i, src in enumerate(sources):
        hypotheses.append(
            generate(model, tokenizer, src, device, strategy=strategy,
                     max_new_tokens=max_new_tokens)
        )
        if (i + 1) % 100 == 0 or i + 1 == len(sources):
            print(f"  translated {i + 1}/{len(sources)}", file=sys.stderr)
    return hypotheses


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--limit", type=int, default=None,
                        help=f"score only the first N val pairs (default: all {N_VAL})")
    parser.add_argument("--strategy", choices=["greedy", "topk", "topp"],
                        default="greedy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    source_texts, target_texts, val_idx, _ = load_opus_splits()
    val_sources = [source_texts[i] for i in val_idx]
    val_refs = [target_texts[i] for i in val_idx]

    print(f"translating {len(val_sources[:args.limit] if args.limit else val_sources)} "
          f"val pairs on {device.type} ...", file=sys.stderr)
    hypotheses = translate_sources(model, tokenizer, val_sources, device,
                                   strategy=args.strategy, limit=args.limit)
    references = val_refs[:len(hypotheses)]

    bleu, signature = corpus_bleu(hypotheses, references)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    print(f"BLEU {bleu:.2f} | n={len(hypotheses)} | "
          f"strategy={args.strategy} | ckpt epoch {ckpt['epoch']}")
    print(signature)

    for hyp, ref in list(zip(hypotheses, references))[:5]:
        print(f"  src → hyp: {hyp}")
        print(f"       ref : {ref}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    metrics = {
        "bleu": bleu,
        "signature": signature,
        "n": len(hypotheses),
        "strategy": args.strategy,
        "checkpoint": args.checkpoint,
        "epoch": ckpt["epoch"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
