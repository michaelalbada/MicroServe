"""Reference generation loops used to validate later serving optimizations."""

from __future__ import annotations

import torch

from microserve.model import Transformer


def _next_token(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1].argmax(dim=-1, keepdim=True)


@torch.inference_mode()
def generate_naive(
    model: Transformer, prompt: torch.Tensor, *, max_new_tokens: int
) -> torch.Tensor:
    """Generate greedily by recomputing the complete sequence at every step."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    tokens = prompt
    for _ in range(max_new_tokens):
        tokens = torch.cat((tokens, _next_token(model(tokens))), dim=1)
    return tokens


@torch.inference_mode()
def generate_cached(
    model: Transformer, prompt: torch.Tensor, *, max_new_tokens: int
) -> torch.Tensor:
    """Prefill once, then decode one token at a time from a KV cache."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if prompt.ndim != 2 or prompt.size(1) == 0:
        raise ValueError("prompt must have shape [batch, tokens] with tokens > 0")
    if max_new_tokens == 0:
        return prompt

    total_tokens = prompt.size(1) + max_new_tokens
    cache = model.new_cache(batch_size=prompt.size(0), max_tokens=total_tokens)
    logits = model(prompt, cache=cache)
    tokens = prompt

    for step in range(max_new_tokens):
        token = _next_token(logits)
        tokens = torch.cat((tokens, token), dim=1)
        if step + 1 < max_new_tokens:
            logits = model(token, cache=cache)
    return tokens
