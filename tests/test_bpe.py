import torch

from tokenizer import (BYTE_CHARS, PAD_TOKEN, apply_merge, byte_to_unicode,
                       pre_tokenize, BPETokenizer)
from training import PAD_ID

CORPUS = ["hello world", "ola mundo", "the cat sat", "hello ola"]


def test_special_token_ids(tokenizer):
    assert tokenizer.pad_id == PAD_ID == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3
    assert tokenizer.vocab[:4] == [PAD_TOKEN, "<unk>", "<bos>", "<eos>"]


def test_base_vocab_is_the_256_byte_alphabet():
    tok = BPETokenizer.train(CORPUS, num_merges=0)
    assert len(tok) == 4 + 256
    assert BYTE_CHARS == [byte_to_unicode()[b] for b in sorted(byte_to_unicode())]
    assert len(set(BYTE_CHARS)) == 256


def test_pre_tokenize_uses_gpt2_pattern():
    chunks = pre_tokenize("Hello, world")
    assert "Hello" in chunks and "," in chunks
    assert "Ġworld" in chunks  # byte-level: leading space becomes Ġ


def test_encode_wraps_with_bos_eos(tokenizer):
    ids = tokenizer.encode("hello world")
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_roundtrip_exact(tokenizer):
    for text in ["hello world", "The cat sat.", "ola mundo"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_accents_and_emoji(tokenizer):
    # byte-level BPE never needs <unk>: unseen characters encode as raw bytes
    text = "Olá! café 😀"
    ids = tokenizer.encode(text)
    assert tokenizer.unk_id not in ids
    assert tokenizer.decode(ids) == text


def test_padding_fills_to_max_len(tokenizer):
    ids = tokenizer.encode("hello", max_len=16)
    assert len(ids) == 16
    assert ids[-1] == tokenizer.pad_id


def test_truncation_keeps_eos(tokenizer):
    ids = tokenizer.encode("hello world ola mundo the cat sat", max_len=5)
    assert len(ids) == 5
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_decode_skips_specials(tokenizer):
    out = tokenizer.decode(tokenizer.encode("hello world"))
    for special in ("<bos>", "<eos>", "<pad>", "<unk>"):
        assert special not in out


def test_apply_merge_is_left_to_right():
    assert apply_merge(["h", "e", "l", "l", "o"], "l", "l") == ["h", "e", "ll", "o"]
    assert apply_merge(["a", "a", "a"], "a", "a") == ["aa", "a"]
    assert apply_merge(["x", "y"], "a", "b") == ["x", "y"]


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, num_merges=50)
    b = BPETokenizer.train(CORPUS, num_merges=50)
    assert a.vocab == b.vocab
    assert a.merges == b.merges


def test_state_roundtrip(tokenizer):
    restored = BPETokenizer.from_state(tokenizer.state())
    text = "hello world"
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.decode(restored.encode(text)) == tokenizer.decode(tokenizer.encode(text))


def test_state_is_torch_weights_only_safe(tokenizer):
    import io
    buf = io.BytesIO()
    torch.save(tokenizer.state(), buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=True)
    assert BPETokenizer.from_state(state).vocab == tokenizer.vocab


def test_len_matches_vocab(tokenizer):
    assert len(tokenizer) == len(tokenizer.vocab)
    assert len(tokenizer.stoi) == len(tokenizer)


def test_encode_output_is_int_list(tokenizer):
    ids = tokenizer.encode("the cat sat")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert torch.tensor(ids).dtype == torch.long
