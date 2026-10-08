import torch

from generate import benchmark, generate


def test_greedy_cached_matches_uncached(tiny_model, tokenizer):
    device = torch.device("cpu")
    cached = generate(tiny_model, tokenizer, "hello world", device,
                      strategy="greedy", use_cache=True)
    uncached = generate(tiny_model, tokenizer, "hello world", device,
                        strategy="greedy", use_cache=False)
    assert cached == uncached


def test_generate_returns_string(tiny_model, tokenizer):
    out = generate(tiny_model, tokenizer, "hello", torch.device("cpu"),
                   strategy="greedy", max_new_tokens=4)
    assert isinstance(out, str)
    assert "<bos>" not in out and "<pad>" not in out


def test_max_new_tokens_respected(tiny_model, tokenizer):
    device = torch.device("cpu")
    # tiny random model may emit EOS; force no stop and count tokens
    text = generate(tiny_model, tokenizer, "the cat sat", device,
                    strategy="greedy", max_new_tokens=5, stop_on_eos=False)
    assert len(text.split()) <= 5


def test_cache_cleared_between_calls(tiny_model, tokenizer):
    device = torch.device("cpu")
    generate(tiny_model, tokenizer, "hello", device, use_cache=True)
    for layer in tiny_model.decoder.dec_layers:
        assert layer.self_attention.kv_cache.key is None


def test_benchmark_runs(tiny_model, tokenizer):
    results = benchmark(tiny_model, tokenizer, "hello", torch.device("cpu"),
                        n_tokens=4, repeats=1)
    assert set(results) == {False, True}
    assert results[True] > 0 and results[False] > 0


def test_sampling_strategies_deterministic_with_seed(tiny_model, tokenizer):
    device = torch.device("cpu")
    torch.manual_seed(0)
    a = generate(tiny_model, tokenizer, "hello", device, strategy="topk",
                 top_k=5, max_new_tokens=5, stop_on_eos=False)
    torch.manual_seed(0)
    b = generate(tiny_model, tokenizer, "hello", device, strategy="topk",
                 top_k=5, max_new_tokens=5, stop_on_eos=False)
    assert a == b
