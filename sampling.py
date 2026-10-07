import torch


def _prepare_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Applies temperature to the logits"""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature != 1.0:
        logits = logits / temperature
    return logits


def _sample(logits: torch.Tensor) -> torch.Tensor:
    """Multinomial sample one token per row.
    """
    probs = torch.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, num_samples=1)
    return idx.reshape(probs.shape[:-1])


def top_k_generation(logits: torch.Tensor, top_k: int = 50,
                     return_logits: bool = False, min_tokens_to_keep: int = 1,
                     temperature: float = 1.0) -> torch.Tensor:
    logits = _prepare_probs(logits, temperature)
    top_k = max(min(top_k, logits.shape[-1]), min_tokens_to_keep)

    top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
    threshold = top_k_logits[..., -1:]
    masked_logits = logits.masked_fill(logits < threshold, float('-inf'))

    sample_token = _sample(masked_logits)

    if return_logits:
        return sample_token, masked_logits
    return sample_token


def top_p_generation(logits: torch.Tensor, top_p: float = 0.9,
                     return_logits: bool = False, min_tokens_to_keep: int = 1,
                     temperature: float = 1.0) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    logits = _prepare_probs(logits, temperature)

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Drop the smallest tokens until the remaining tail carries `top_p` mass
    indices_to_remove = cum_probs <= 1 - top_p
    indices_to_remove[..., -min_tokens_to_keep:] = False
    indices_to_remove = indices_to_remove.scatter(-1, sorted_indices, indices_to_remove)

    top_p_logits = logits.masked_fill(indices_to_remove, float('-inf'))

    sample_token = _sample(top_p_logits)

    if return_logits:
        return sample_token, top_p_logits
    return sample_token