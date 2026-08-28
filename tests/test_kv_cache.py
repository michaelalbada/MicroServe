import pytest
import torch

from microserve import KVCache


def make_cache(max_tokens: int = 4) -> KVCache:
    return KVCache(
        num_layers=2,
        batch_size=1,
        max_tokens=max_tokens,
        num_kv_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_append_exposes_active_prefix() -> None:
    cache = make_cache()
    first = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    keys, values = cache.append(0, first, first + 10, start=0)
    cache.advance(2)
    second = torch.tensor([[[[5.0, 6.0]]]])
    keys, values = cache.append(0, second, second + 10, start=2)

    torch.testing.assert_close(keys, torch.cat((first, second), dim=1))
    torch.testing.assert_close(values, torch.cat((first + 10, second + 10), dim=1))


def test_capacity_overflow_is_explicit() -> None:
    cache = make_cache(max_tokens=1)
    states = torch.zeros(1, 2, 1, 2)

    with pytest.raises(ValueError, match="capacity exceeded"):
        cache.append(0, states, states, start=0)


def test_reset_reuses_allocation() -> None:
    cache = make_cache()
    keys_address = cache.keys.data_ptr()
    cache.advance(3)

    cache.reset()

    assert cache.length == 0
    assert cache.keys.data_ptr() == keys_address
