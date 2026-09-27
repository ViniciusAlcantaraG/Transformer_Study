from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn, optim
from transformer import Transformer

class SimpleTokenizer:
    def __init__(self, sentences):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for text in sentences:
            vocab.extend(text.lower().split())

        self.vocab = sorted(set(vocab))
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.pad_id = self.stoi[self.pad_token]
        self.unk_id = self.stoi[self.unk_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]

    def encode(self, text, max_len=None):
        tokens = [self.bos_token] + text.lower().split() + [self.eos_token]
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokens]
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.vocab)

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=16):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = torch.tensor(self.tokenizer.encode(self.src_texts[idx], self.max_len), dtype=torch.long)
        tgt = torch.tensor(self.tokenizer.encode(self.tgt_texts[idx], self.max_len), dtype=torch.long)
        return src, tgt

def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    max_len = max(max(len(x), len(y)) for x, y in zip(src_batch, tgt_batch))
    padded_src, padded_tgt = [], []
    for src, tgt in zip(src_batch, tgt_batch):
        src = src.tolist()
        tgt = tgt.tolist()
        src = src + [0] * (max_len - len(src))
        tgt = tgt + [0] * (max_len - len(tgt))
        padded_src.append(src)
        padded_tgt.append(tgt)
    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_tgt, dtype=torch.long)

def causal_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

ds = load_dataset("Helsinki-NLP/opus-100", "en-pt", split="train[:500]")
source_texts = [ex["translation"]["en"] for ex in ds]
target_texts = [ex["translation"]["pt"] for ex in ds]

tokenizer = SimpleTokenizer(source_texts + target_texts)
vocab_size = len(tokenizer)

dataset = TranslationDataset(source_texts, target_texts, tokenizer, max_len=16)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    d_model=128,
    dff=512,
    num_layers=2,
    num_heads=4,
    vocab_size=vocab_size,
    max_position_encod=16,
    p_rate=0.1,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total = 0.0

    for src_batch, tgt_batch in loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        look_ahead_mask = causal_mask(tgt_batch.size(1)).unsqueeze(0).unsqueeze(0).to(device)
        padding_mask = (src_batch == 0).unsqueeze(1).unsqueeze(2).to(device)

        logits = model(src_batch, tgt_batch, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_batch.reshape(-1))
        loss.backward()
        optimizer.step()

        total += loss.item()

    print(f"Epoch {epoch}: {total/len(loader):.4f}")