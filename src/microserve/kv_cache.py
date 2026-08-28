"""A deliberately simple, contiguous KV cache.

This is stage 1 of the serving curriculum. It makes the memory cost and append
operation explicit; paged allocation will replace the contiguous layout later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Cache(Protocol):
    """The small cache contract consumed by the model."""

    length: int

    @property
    def batch_size(self) -> int: ...

    @property
    def capacity(self) -> int: ...

    def append(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def advance(self, tokens: int) -> None: ...

    def truncate(self, length: int) -> None: ...

    def close(self) -> None: ...


class KVCache:
    """Preallocated key/value storage shared by all transformer layers.

    The first implementation advances every sequence in the batch together.
    That limitation is useful: continuous batching will have to replace this
    single cursor with per-request state rather than hiding the problem here.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        max_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (num_layers, batch_size, max_tokens, num_kv_heads, head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty(shape, device=device, dtype=dtype)
        self.length = 0

    @property
    def batch_size(self) -> int:
        return self.keys.size(1)

    @property
    def capacity(self) -> int:
        return self.keys.size(2)

    def append(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store one layer's new states and return its complete active prefix."""
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        if keys.ndim != 4:
            raise ValueError("expected [batch, tokens, kv_heads, head_dim]")
        if keys.size(0) != self.batch_size:
            raise ValueError(
                f"cache batch size is {self.batch_size}, received {keys.size(0)}"
            )
        if start != self.length:
            raise ValueError(f"cache cursor is {self.length}, received start={start}")

        end = start + keys.size(1)
        if end > self.capacity:
            raise ValueError(
                f"KV cache capacity exceeded: need {end} tokens, have {self.capacity}"
            )

        self.keys[layer, :, start:end].copy_(keys)
        self.values[layer, :, start:end].copy_(values)
        return self.keys[layer, :, :end], self.values[layer, :, :end]

    def advance(self, tokens: int) -> None:
        """Commit a model step after every layer has appended its states."""
        if tokens < 0 or self.length + tokens > self.capacity:
            raise ValueError("invalid KV cache advance")
        self.length += tokens

    def reset(self) -> None:
        """Make the allocation reusable without clearing overwritten storage."""
        self.length = 0

    def truncate(self, length: int) -> None:
        """Roll back speculative tokens without clearing their stale values."""
        if length < 0 or length > self.length:
            raise ValueError("cache can only truncate its active prefix")
        self.length = length

    def close(self) -> None:
        """Contiguous storage is owned by this object and needs no release."""
