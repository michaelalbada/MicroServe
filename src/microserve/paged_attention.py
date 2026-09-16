"""Reference PagedAttention over indirect KV block tables.

The implementation is intentionally written in ordinary PyTorch.  It is not a
fused performance kernel; it is the correctness oracle such a kernel should
match.  Its important systems property is real: keys and values stay in their
physical pages, and attention visits those pages through each request's block
table without reconstructing contiguous KV or padding ragged requests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from microserve.kv_cache import PagedKVView


def paged_attention(
    query: torch.Tensor,
    views: Sequence[PagedKVView],
) -> torch.Tensor:
    """Attend ``[batch, query_tokens, heads, head_dim]`` over paged KV.

    Each view may have a different logical length and physical block table.
    Blocks are combined with an online softmax, so temporary storage is bounded
    by one KV block rather than the longest request in the batch.
    """
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, tokens, heads, head_dim]")
    if query.size(0) != len(views) or not views:
        raise ValueError("one paged KV view is required for each query row")

    rows = []
    for row, view in enumerate(views):
        rows.append(_paged_attention_row(query[row], view))
    return torch.stack(rows)


def _paged_attention_row(
    query: torch.Tensor,
    view: PagedKVView,
) -> torch.Tensor:
    query_tokens, num_heads, head_dim = query.shape
    if head_dim != view.head_dim:
        raise ValueError("query and KV head dimensions differ")
    if num_heads % view.num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if query_tokens > view.length:
        raise ValueError("query tokens cannot exceed the active KV length")

    groups = num_heads // view.num_kv_heads
    # Consecutive query heads share one KV head under grouped-query attention.
    grouped_query = query.reshape(
        query_tokens, view.num_kv_heads, groups, head_dim
    ).float()
    query_start = view.length - query_tokens
    query_positions = torch.arange(
        query_start,
        view.length,
        device=query.device,
        dtype=torch.long,
    )

    accumulator = torch.zeros_like(grouped_query)
    normalizer = torch.zeros(
        query_tokens,
        view.num_kv_heads,
        groups,
        device=query.device,
        dtype=torch.float32,
    )
    maximum = torch.full_like(normalizer, -torch.inf)
    scale = 1 / math.sqrt(head_dim)

    for logical_block, physical_block in enumerate(view.block_table):
        block_start = logical_block * view.block_size
        block_tokens = min(view.block_size, view.length - block_start)
        keys = view.keys[physical_block, :block_tokens].float()
        values = view.values[physical_block, :block_tokens].float()

        scores = torch.einsum("thgd,shd->thgs", grouped_query, keys) * scale
        key_positions = torch.arange(
            block_start,
            block_start + block_tokens,
            device=query.device,
            dtype=torch.long,
        )
        visible = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~visible[:, None, None, :], -torch.inf)

        block_maximum = scores.amax(dim=-1)
        new_maximum = torch.maximum(maximum, block_maximum)
        previous_scale = torch.exp(maximum - new_maximum)
        weights = torch.exp(scores - new_maximum[..., None])

        accumulator = accumulator * previous_scale[..., None]
        accumulator = accumulator + torch.einsum(
            "thgs,shd->thgd", weights, values
        )
        normalizer = normalizer * previous_scale + weights.sum(dim=-1)
        maximum = new_maximum

    attended = accumulator / normalizer[..., None]
    return attended.reshape(query_tokens, num_heads, head_dim).to(query.dtype)
