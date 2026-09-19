import torch
from torch import nn

def top_k_generation(logits, top_k=50, return_logits=False, min_tokens_to_keep=1):

    top_k = max(top_k, min_tokens_to_keep)
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    
    indices_to_remove = logits < top_k_logits[...,-1]
    masked_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    masked_prob = torch.softmax(masked_logits, dim=-1)

    # sampling
    sample_token = torch.multinomial(masked_prob, num_samples=1)[0]

    if return_logits:
        return sample_token, masked_logits
    return sample_token


def top_p_generation(logits, top_p=0.9, return_logits=False, min_tokens_to_keep=1):

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    indices_to_remove = cumum_probs <= 1 - top_p 
    indices_to_remove[..., -min_tokens_to_keep:] = False
    indices_to_remove = indices_to_remove.scatter(-1, sorted_indices, indices_to_remove)

    top_p_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    top_p_prob = torch.softmax(top_p_logits, dim=-1)

    # sampling
    sample_token = torch.multinomial(top_p_prob, num_samples=1)[0]

    if return_logits:
        return sample_token, top_p_logits
    return sample_token