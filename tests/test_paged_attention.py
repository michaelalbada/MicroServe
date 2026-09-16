import torch
import torch.nn.functional as F

from microserve import PagedKVView, paged_attention


def _contiguous(view: PagedKVView) -> tuple[torch.Tensor, torch.Tensor]:
    keys = []
    values = []
    remaining = view.length
    for block in view.block_table:
        tokens = min(remaining, view.block_size)
        keys.append(view.keys[block, :tokens])
        values.append(view.values[block, :tokens])
        remaining -= tokens
    return torch.cat(keys), torch.cat(values)


def _expected(query: torch.Tensor, view: PagedKVView) -> torch.Tensor:
    keys, values = _contiguous(view)
    groups = query.size(1) // view.num_kv_heads
    keys = keys.repeat_interleave(groups, dim=1)
    values = values.repeat_interleave(groups, dim=1)
    query_positions = torch.arange(view.length - query.size(0), view.length)
    key_positions = torch.arange(view.length)
    visible = key_positions[None, :] <= query_positions[:, None]
    attended = F.scaled_dot_product_attention(
        query.transpose(0, 1)[None],
        keys.transpose(0, 1)[None],
        values.transpose(0, 1)[None],
        attn_mask=visible[None, None],
    )
    return attended[0].transpose(0, 1)


def test_paged_attention_follows_noncontiguous_block_tables_for_ragged_decode() -> None:
    torch.manual_seed(23)
    keys = torch.randn(7, 2, 2, 4)
    values = torch.randn_like(keys)
    views = [
        PagedKVView(keys, values, (4, 1, 6), length=5),
        PagedKVView(keys, values, (3, 0), length=3),
    ]
    query = torch.randn(2, 1, 4, 4)

    actual = paged_attention(query, views)
    expected = torch.stack(
        [_expected(query[row], view) for row, view in enumerate(views)]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_paged_attention_applies_causality_across_prefill_blocks() -> None:
    torch.manual_seed(29)
    keys = torch.randn(6, 2, 1, 8)
    values = torch.randn_like(keys)
    view = PagedKVView(keys, values, (5, 2, 4), length=5)
    query = torch.randn(1, 3, 4, 8)

    actual = paged_attention(query, [view])
    expected = _expected(query[0], view)[None]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
