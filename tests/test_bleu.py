import torch

from evaluate import corpus_bleu, translate_sources


def test_exact_match_scores_100():
    score, _ = corpus_bleu(["bom dia mundo como estas"],
                           ["bom dia mundo como estas"])
    assert abs(score - 100.0) < 1e-6


def test_empty_hypothesis_scores_zero():
    score, _ = corpus_bleu([""], ["bom dia"])
    assert score == 0.0


def test_no_overlap_scores_zero():
    score, _ = corpus_bleu(["xyz abc def ghi"], ["bom dia mundo"])
    assert score == 0.0


def test_score_in_unit_interval_and_deterministic():
    hyps = ["bom dia mundo", "amo te"]
    refs = ["bom dia amigo", "amo te"]
    score_a, _ = corpus_bleu(hyps, refs)
    score_b, _ = corpus_bleu(hyps, refs)
    assert 0.0 <= score_a <= 100.0
    assert score_a == score_b


def test_signature_reports_13a_tokenization():
    _, signature = corpus_bleu(["bom dia"], ["bom dia"])
    assert "tok:13a" in signature
    assert "nrefs:1" in signature


def test_translate_sources_returns_one_string_per_source(tiny_model, tokenizer):
    hyps = translate_sources(tiny_model, tokenizer, ["hello", "the cat"],
                             torch.device("cpu"), max_new_tokens=4)
    assert len(hyps) == 2
    assert all(isinstance(h, str) for h in hyps)
    assert all("<bos>" not in h and "<pad>" not in h for h in hyps)


def test_translate_sources_respects_limit(tiny_model, tokenizer):
    hyps = translate_sources(tiny_model, tokenizer, ["hello", "the cat", "ola"],
                             torch.device("cpu"), max_new_tokens=4, limit=2)
    assert len(hyps) == 2
