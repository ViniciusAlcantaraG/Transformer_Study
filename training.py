import json
import math
from pathlib import Path

from datasets import load_dataset
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset

from tokenizer import BPETokenizer
from transformer import Transformer

PAD_ID = 0

# training hyperparameters
TRAIN_PAIRS = 20000
N_VAL = 1000
# source/target sequences are truncated + padded to this max_len
MAX_LEN = 32          
BATCH_SIZE = 32
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
# 4 specials + 256 byte alphabet + this many merges 
BPE_NUM_MERGES = 4000
LABEL_SMOOTHING = 0.1


class TranslationDataset(Dataset):
    def __init__(self, src_texts: list[str], tgt_texts: list[str],
                 tokenizer: BPETokenizer, max_len: int = 32) -> None:
        # Encode once instead of re-tokenizing every sample on every epoch
        self.src = [torch.tensor(tokenizer.encode(t, max_len), dtype=torch.long)
                    for t in src_texts]
        self.tgt = [torch.tensor(tokenizer.encode(t, max_len), dtype=torch.long)
                    for t in tgt_texts]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.src[idx], self.tgt[idx]

def collate_batch(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    src_batch, tgt_batch = zip(*batch)
    max_len = max(max(len(x), len(y)) for x, y in zip(src_batch, tgt_batch))
    padded_src, padded_tgt = [], []
    for src, tgt in zip(src_batch, tgt_batch):
        src = src.tolist()
        tgt = tgt.tolist()
        src = src + [PAD_ID] * (max_len - len(src))
        tgt = tgt + [PAD_ID] * (max_len - len(tgt))
        padded_src.append(src)
        padded_tgt.append(tgt)
    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_tgt, dtype=torch.long)


def causal_mask(seq_len: int) -> Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


def make_masks(src_batch: Tensor, tgt_batch: Tensor,
               pad_id: int = PAD_ID) -> tuple[Tensor, Tensor, Tensor]:
    """Build the three masks used by the model.

    Returns:
        encoder_mask:      (B, 1, 1, S) bool, True = masked. Hides source padding
                           from the encoder's self-attention.
        look_ahead_mask:   (B, 1, T, T) bool, True = masked. Combines the causal
                           (no looking ahead) constraint with target padding.
        cross_padding_mask:(B, 1, 1, S) bool, True = masked. Hides source padding
                           from the decoder's cross-attention over encoder states.
    """
    src_pad = (src_batch == pad_id).unsqueeze(1).unsqueeze(2)
    tgt_pad = (tgt_batch == pad_id).unsqueeze(1).unsqueeze(2)
    causal = causal_mask(tgt_batch.size(1)).unsqueeze(0).unsqueeze(0).to(tgt_batch.device)
    look_ahead = causal | tgt_pad
    return src_pad, look_ahead, src_pad


def build_model_config(vocab_size: int, max_len: int = MAX_LEN) -> dict[str, float | int]:
    return {
        "d_model": 256,
        "dff": 1024,
        "num_layers": 4,
        "num_heads": 4,
        "vocab_size": vocab_size,
        "max_position_encod": max_len,
        "p_rate": 0.1,
    }


def shift_targets(tgt_batch: Tensor) -> tuple[Tensor, Tensor]:
    """Split a target sequence into decoder input and labels.

    Next-token prediction: given [bos, w1, w2, eos] the decoder input is
    [bos, w1, w2] and the labels are [w1, w2, eos].
    """
    return tgt_batch[:, :-1], tgt_batch[:, 1:]


def evaluate(model: Transformer, loader: DataLoader, criterion: nn.Module,
             device: torch.device, pad_id: int = PAD_ID) -> tuple[float, float]:
    """Token-weighted mean cross-entropy and perplexity."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            dec_in, labels = shift_targets(tgt_batch)
            encoder_mask, look_ahead, cross_mask = make_masks(src_batch, dec_in)
            logits = model(src_batch, dec_in, mask=encoder_mask,
                           look_ahead_mask=look_ahead, padding_mask=cross_mask)
            loss = criterion(logits.reshape(-1, model.vocab_size), labels.reshape(-1))
            n_tokens = int((labels != pad_id).sum())
            total += loss.item() * n_tokens
            count += n_tokens
    mean_loss = total / max(count, 1)
    return mean_loss, math.exp(mean_loss)


def plot_history(history: dict[str, list[float]], path: str | Path) -> None:
    """Save train/val loss curves to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train"], marker="o", label="train loss")
    plt.plot(epochs, history["val"], marker="o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    ds = load_dataset("Helsinki-NLP/opus-100", "en-pt", split=f"train[:{TRAIN_PAIRS}]")
    source_texts = [ex["translation"]["en"] for ex in ds]
    target_texts = [ex["translation"]["pt"] for ex in ds]

    torch.manual_seed(42)

    # random train/val split
    perm = torch.randperm(len(source_texts)).tolist()
    val_idx, train_idx = perm[:N_VAL], perm[N_VAL:]

    # learn BPE merges on the train split only
    # byte-level encoding means the validation split always encodes anyway
    tokenizer = BPETokenizer.train(
        [source_texts[i] for i in train_idx] + [target_texts[i] for i in train_idx],
        num_merges=BPE_NUM_MERGES,
    )
    assert tokenizer.pad_id == PAD_ID, "PAD_ID constant must match the tokenizer"
    vocab_size = len(tokenizer)
    print(f"{len(source_texts)} pairs | vocab {vocab_size} "
          f"({len(tokenizer.merges)} merges) | max_len {MAX_LEN}")

    train_set = TranslationDataset([source_texts[i] for i in train_idx],
                                   [target_texts[i] for i in train_idx],
                                   tokenizer, max_len=MAX_LEN)
    val_set = TranslationDataset([source_texts[i] for i in val_idx],
                                 [target_texts[i] for i in val_idx],
                                 tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = build_model_config(vocab_size, max_len=MAX_LEN)
    model = Transformer(**model_config).to(device)
    print(f"device {device.type} | params {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id,
                                    label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # linear warmup, then inverse square-root decay
    warmup_steps = 500

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return (warmup_steps / max(step, 1)) ** 0.5

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history: dict[str, list[float]] = {"train": [], "val": [], "val_ppl": []}
    best_val = float("inf")
    epochs_no_improve = 0
    out_dir = Path("results")
    ckpt_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0

        for src_batch, tgt_batch in train_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()

            dec_in, labels = shift_targets(tgt_batch)
            encoder_mask, look_ahead_mask, cross_padding_mask = make_masks(src_batch, dec_in)

            logits = model(
                src_batch,
                dec_in,
                mask=encoder_mask,
                look_ahead_mask=look_ahead_mask,
                padding_mask=cross_padding_mask,
            )
            loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            n_tokens = int((labels != tokenizer.pad_id).sum())
            total += loss.item() * n_tokens
            count += n_tokens

        train_loss = total / max(count, 1)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d}: train {train_loss:.4f} | val {val_loss:.4f} "
              f"| ppl {val_ppl:.2f} | lr {lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model_config,
                "tokenizer": tokenizer.state(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"early stopping after {epoch} epochs "
                      f"({epochs_no_improve} without improvement)")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out_dir / "loss_curve.png")
    print(f"best val loss {best_val:.4f} | saved checkpoints/best.pt, "
          f"results/history.json, results/loss_curve.png")


if __name__ == "__main__":
    main()
