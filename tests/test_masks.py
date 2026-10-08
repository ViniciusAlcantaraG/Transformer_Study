import torch

from training import PAD_ID, causal_mask, make_masks


def test_causal_mask_blocks_future_only():
    mask = causal_mask(4)
    # True = masked
    assert bool(mask[0, 1])
    assert not bool(mask[0, 0])
    assert not bool(mask[3, 3])
    # strictly upper triangular
    assert int(mask.sum()) == 4 * 3 // 2
    assert not bool(torch.tril(mask, diagonal=0).any())


def test_mask_shapes(batches):
    src, tgt = batches
    enc, look, cross = make_masks(src, tgt)
    assert enc.shape == (2, 1, 1, 4)
    assert look.shape == (2, 1, 4, 4)
    assert cross.shape == (2, 1, 1, 4)


def test_encoder_mask_hides_source_padding(batches):
    src, tgt = batches
    enc, _, _ = make_masks(src, tgt)
    for b in range(src.shape[0]):
        for s in range(src.shape[1]):
            assert bool(enc[b, 0, 0, s]) == (src[b, s] == PAD_ID)


def test_look_ahead_is_causal_plus_target_padding(batches):
    src, tgt = batches
    _, look, _ = make_masks(src, tgt)
    causal = causal_mask(tgt.shape[1])
    for b in range(tgt.shape[0]):
        for t in range(tgt.shape[1]):
            row = look[b, 0, t]
            expected = causal[t] | (tgt[b] == PAD_ID)
            assert torch.equal(row, expected)


def test_custom_pad_id():
    src = torch.tensor([[4, 5, 9]])
    tgt = torch.tensor([[4, 6, 9]])
    enc, _, _ = make_masks(src, tgt, pad_id=9)
    assert not enc[0, 0, 0, 0] and enc[0, 0, 0, 2]
