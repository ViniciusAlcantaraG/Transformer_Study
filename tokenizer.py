"""From-scratch byte-level BPE tokenizer built on the GPT-2 pre-tokenizer.

Study implementation (following the Hugging Face course) completed with the
three pieces that make it usable for real training runs:

1. byte-level alphabet — the 256 GPT-2 byte-to-unicode mappings form the base
   vocabulary, so *any* input text encodes (out-of-vocabulary is impossible);
2. incremental pair bookkeeping — merges are learned by updating only the
   words touched by the last merge, instead of rescanning the corpus;
3. a real `encode`/`decode` plus checkpoint-serializable `state()`.

Run this file directly for the original 4-sentence demo.
"""

from __future__ import annotations

import sys
from collections import Counter
from functools import lru_cache

from transformers import AutoTokenizer

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIALS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


@lru_cache(maxsize=1)
def gpt2_pretokenizer():
    """
    We get the GPT-2 tokenizer from the transformers lib
    """
    return AutoTokenizer.from_pretrained("gpt2").backend_tokenizer.pre_tokenizer


def pre_tokenize(text: str) -> list[str]:
    """Split text into GPT-2 chunks."""
    return [chunk for chunk, _ in gpt2_pretokenizer().pre_tokenize_str(text)]


def byte_to_unicode() -> dict[int, str]:
    """GPT-2's bytes_to_unicode()
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_MAP = byte_to_unicode()
BYTE_CHARS = [BYTE_MAP[b] for b in sorted(BYTE_MAP)]          # alphabet, byte order
UNICODE_TO_BYTE = {ch: b for b, ch in BYTE_MAP.items()}        # reverse map for decode


def apply_merge(split: list[str], a: str, b: str) -> list[str]:
    """Merge every non-overlapping adjacent (a, b) occurrence, left to right."""
    out: list[str] = []
    i = 0
    while i < len(split):
        if i + 1 < len(split) and split[i] == a and split[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(split[i])
            i += 1
    return out


def _adjust(counter: Counter, pair: tuple[str, str], delta: int) -> None:
    """Add `delta` to a pair count, deleting the key when it reaches <= 0."""
    value = counter.get(pair, 0) + delta
    if value > 0:
        counter[pair] = value
    else:
        counter.pop(pair, None)


class BPETokenizer:
    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]) -> None:
        self.vocab = list(vocab)
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

    def __len__(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, sentences, num_merges: int = 4000) -> "BPETokenizer":
        """Learn `num_merges` BPE merges from `sentences`.

        Pair frequencies are maintained incrementally: merging (a, b) only
        touches the words that contain that pair, so a full run over a 19k
        sentence corpus stays well under a minute.
        """
        word_freqs: Counter = Counter()
        for text in sentences:
            word_freqs.update(pre_tokenize(str(text)))

        splits: dict[str, list[str]] = {w: list(w) for w in word_freqs}
        vocab: list[str] = list(SPECIALS) + BYTE_CHARS
        vocab_set = set(vocab)
        merges: list[tuple[str, str]] = []

        pair_freqs: Counter = Counter()
        pair_words: dict[tuple[str, str], set[str]] = {}
        for w, split in splits.items():
            freq = word_freqs[w]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
                pair_words.setdefault(pair, set()).add(w)

        for _ in range(num_merges):
            if not pair_freqs:
                break
            # most frequent pair
            # ties broken in alphabetical order
            best = min(pair_freqs, key=lambda p: (-pair_freqs[p], p))
            a, b = best
            merged = a + b
            if merged in vocab_set:
                # merging would collide with an existing token
                pair_freqs.pop(best, None)
                pair_words.pop(best, None)
                continue

            for w in list(pair_words.get(best, ())):
                old = splits[w]
                new = apply_merge(old, a, b)
                if new == old:
                    continue
                freq = word_freqs[w]
                for i in range(len(old) - 1):
                    pair = (old[i], old[i + 1])
                    _adjust(pair_freqs, pair, -freq)
                    words = pair_words.get(pair)
                    if words is not None:
                        words.discard(w)
                        if not words:
                            del pair_words[pair]
                splits[w] = new
                for i in range(len(new) - 1):
                    pair = (new[i], new[i + 1])
                    pair_freqs[pair] += freq
                    pair_words.setdefault(pair, set()).add(w)

            pair_freqs.pop(best, None)
            pair_words.pop(best, None)
            vocab.append(merged)
            vocab_set.add(merged)
            merges.append((a, b))

        return cls(vocab, merges)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [self.bos_id]
        for chunk in pre_tokenize(text):
            tokens = list(chunk)
            while len(tokens) > 1:
                best: tuple[int, int] | None = None
                for i in range(len(tokens) - 1):
                    rank = self.ranks.get((tokens[i], tokens[i + 1]))
                    if rank is not None and (best is None or rank < best[0]):
                        best = (rank, i)
                if best is None:
                    break
                i = best[1]
                tokens = tokens[:i] + [tokens[i] + tokens[i + 1]] + tokens[i + 2:]
            ids.extend(self.stoi.get(tok, self.unk_id) for tok in tokens)
        ids.append(self.eos_id)
        if max_len is not None:
            if len(ids) > max_len:
                # Truncate but always keep the final <eos>
                ids = ids[:max_len - 1] + [self.eos_id]
            else:
                ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    # ----------------------------------------------------------------- decode

    def decode(self, ids) -> str:
        """Byte-level decode: join token strings, reverse the byte map, UTF-8.

        Specials (`<pad>`, `<bos>`, `<eos>`, `<unk>`) are skipped, so the
        output is plain text with correct spacing and punctuation.
        """
        skip = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        pieces = [self.itos.get(i, "") for i in ids if i not in skip]
        data = bytes(UNICODE_TO_BYTE[ch] for ch in "".join(pieces)
                     if ch in UNICODE_TO_BYTE)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- state

    def state(self) -> dict:
        """Plain-Python checkpoint payload (safe for torch.load weights_only)."""
        return {"vocab": list(self.vocab),
                "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        return cls(state["vocab"], [tuple(m) for m in state["merges"]])


def _demo() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    tok = BPETokenizer.train(corpus, num_merges=40)
    print(f"vocab: {len(tok)} tokens (4 specials + 256 bytes + {len(tok.merges)} merges)")
    print("first merges:", tok.merges[:10])
    text = "This course is about tokens."
    ids = tok.encode(text)
    print(f"encode({text!r})")
    print(f"  -> {ids}")
    print(f"  decode -> {tok.decode(ids)!r}")
    assert tok.decode(ids) == text


if __name__ == "__main__":
    _demo()
