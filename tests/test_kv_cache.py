import torch

from training import make_masks


def _full_forward(model, src, tgt):
    """Uncached teacher-forced decode of the complete target sequence."""
    encoder_mask, look_ahead, cross_mask = make_masks(src, tgt)
    with torch.no_grad():
        return model(src, tgt, mask=encoder_mask,
                     look_ahead_mask=look_ahead, padding_mask=cross_mask)


def _cached_decode(model, src, tgt):
    """Token-by-token decode using the KV cache and positional offsets."""
    encoder_mask, _, cross_mask = make_masks(src, tgt)
    model.clear_cache()
    with torch.no_grad():
        encoder_output = model.encoder(src, mask=encoder_mask)
        steps = []
        for t in range(tgt.shape[1]):
            out, _ = model.decoder(
                tgt[:, t:t + 1],
                encoder_output,
                look_ahead_mask=None,
                padding_mask=cross_mask,
                use_cache=True,
                start_pos=t,
            )
            steps.append(model.linear(out))
    logits = torch.cat(steps, dim=1)
    model.clear_cache()
    return logits


def test_cached_decode_matches_full_forward(tiny_model, unpadded_batches):
    src, tgt = unpadded_batches
    full = _full_forward(tiny_model, src, tgt)
    cached = _cached_decode(tiny_model, src, tgt)
    assert cached.shape == full.shape
    assert torch.allclose(cached, full, atol=1e-5), (
        f"max diff {(cached - full).abs().max().item()}"
    )


def test_wrong_start_pos_breaks_equivalence(tiny_model, unpadded_batches):
    # guards the start_pos fix: reusing pe[:1] at every step is not equivalent
    src, tgt = unpadded_batches
    full = _full_forward(tiny_model, src, tgt)
    encoder_mask, _, cross_mask = make_masks(src, tgt)
    tiny_model.clear_cache()
    with torch.no_grad():
        encoder_output = tiny_model.encoder(src, mask=encoder_mask)
        steps = []
        for t in range(tgt.shape[1]):
            out, _ = tiny_model.decoder(
                tgt[:, t:t + 1], encoder_output,
                padding_mask=cross_mask, use_cache=True, start_pos=0,
            )
            steps.append(tiny_model.linear(out))
    wrong = torch.cat(steps, dim=1)
    tiny_model.clear_cache()
    assert not torch.allclose(wrong, full, atol=1e-4)


def test_clear_cache_resets_state(tiny_model, batches):
    src, tgt = batches
    encoder_mask, look_ahead, cross_mask = make_masks(src, tgt)
    with torch.no_grad():
        tiny_model(src, tgt, mask=encoder_mask, look_ahead_mask=look_ahead,
                   padding_mask=cross_mask, use_cache=True)
    layer = tiny_model.decoder.dec_layers[0]
    assert layer.self_attention.kv_cache.key is not None
    tiny_model.clear_cache()
    assert layer.self_attention.kv_cache.key is None
    assert layer.self_attention.kv_cache.value is None


def test_cross_attention_is_never_cached(tiny_model, batches):
    src, tgt = batches
    encoder_mask, look_ahead, cross_mask = make_masks(src, tgt)
    with torch.no_grad():
        tiny_model(src, tgt, mask=encoder_mask, look_ahead_mask=look_ahead,
                   padding_mask=cross_mask, use_cache=True)
    for layer in tiny_model.decoder.dec_layers:
        assert layer.cross_attention.kv_cache.key is None
    tiny_model.clear_cache()


def test_cache_grows_by_one_token_per_step(tiny_model, unpadded_batches):
    src, tgt = unpadded_batches
    encoder_mask, _, cross_mask = make_masks(src, tgt)
    tiny_model.clear_cache()
    with torch.no_grad():
        encoder_output = tiny_model.encoder(src, mask=encoder_mask)
        for t in range(tgt.shape[1]):
            tiny_model.decoder(tgt[:, t:t + 1], encoder_output,
                               padding_mask=cross_mask, use_cache=True,
                               start_pos=t)
            key = tiny_model.decoder.dec_layers[0].self_attention.kv_cache.key
            assert key.shape[2] == t + 1
    tiny_model.clear_cache()
