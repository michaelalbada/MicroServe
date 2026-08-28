"""Stage 3: a physical block allocator for paged KV memory."""

from __future__ import annotations

import math

import torch


class OutOfBlocksError(RuntimeError):
    """The KV pool cannot satisfy an allocation without preemption/eviction."""


class BlockAllocator:
    """Own fixed-size physical KV blocks and reference counts.

    Logical request order lives in each request's block table. Physical blocks
    can therefore be reused in any order and shared by cached prefixes.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if min(num_layers, num_blocks, block_size, num_kv_heads, head_dim) < 1:
            raise ValueError("allocator dimensions must be positive")
        shape = (
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty(shape, device=device, dtype=dtype)
        self.block_size = block_size
        self._free = list(reversed(range(num_blocks)))
        self._references = [0] * num_blocks
        self.peak_used_blocks = 0

    @property
    def num_blocks(self) -> int:
        return self.keys.size(1)

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    @property
    def bytes_per_block(self) -> int:
        # Both K and V are present in every layer.
        return (
            2
            * self.keys.size(0)
            * self.block_size
            * self.keys.size(3)
            * self.keys.size(4)
            * self.keys.element_size()
        )

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.num_blocks

    def allocate(self, count: int = 1) -> list[int]:
        if count < 0:
            raise ValueError("allocation count must be non-negative")
        if count > self.free_blocks:
            raise OutOfBlocksError(
                f"need {count} KV blocks, only {self.free_blocks} are free"
            )
        blocks = [self._free.pop() for _ in range(count)]
        for block in blocks:
            self._references[block] = 1
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)
        return blocks

    def retain(self, blocks: list[int]) -> None:
        for block in blocks:
            if self._references[block] == 0:
                raise ValueError(f"cannot retain free block {block}")
            self._references[block] += 1

    def release(self, blocks: list[int]) -> None:
        for block in blocks:
            if self._references[block] == 0:
                raise ValueError(f"block {block} is already free")
            self._references[block] -= 1
            if self._references[block] == 0:
                self._free.append(block)

    def references(self, block: int) -> int:
        return self._references[block]

    def write(
        self,
        *,
        layer: int,
        block: int,
        offset: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if keys.shape != values.shape or keys.ndim != 4 or keys.size(0) != 1:
            raise ValueError("states must have shape [1, tokens, kv_heads, head_dim]")
        end = offset + keys.size(1)
        if offset < 0 or end > self.block_size:
            raise ValueError("write crosses a physical KV block")
        self.keys[layer, block, offset:end].copy_(keys[0])
        self.values[layer, block, offset:end].copy_(values[0])

    def read(
        self, *, layer: int, blocks: list[int], length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if length < 0 or length > len(blocks) * self.block_size:
            raise ValueError("length is not covered by the logical block table")
        if length == 0:
            shape = (1, 0, self.keys.size(3), self.keys.size(4))
            empty = self.keys.new_empty(shape)
            return empty, empty.clone()

        pieces_k = []
        pieces_v = []
        remaining = length
        for block in blocks:
            used = min(remaining, self.block_size)
            pieces_k.append(self.keys[layer, block, :used])
            pieces_v.append(self.values[layer, block, :used])
            remaining -= used
            if remaining == 0:
                break
        return torch.cat(pieces_k)[None, :], torch.cat(pieces_v)[None, :]

    def blocks_for_tokens(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("token count must be non-negative")
        return math.ceil(tokens / self.block_size)


class PagedKVCache:
    """Map one request's logical token positions onto physical KV blocks."""

    def __init__(self, allocator: BlockAllocator, *, max_tokens: int) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.allocator = allocator
        self.max_tokens = max_tokens
        self.blocks: list[int] = []
        self.length = 0
        self._closed = False

    @classmethod
    def from_shared_blocks(
        cls,
        allocator: BlockAllocator,
        *,
        blocks: list[int],
        length: int,
        max_tokens: int,
    ) -> "PagedKVCache":
        if length % allocator.block_size:
            raise ValueError("only complete physical blocks may be shared")
        if len(blocks) != allocator.blocks_for_tokens(length):
            raise ValueError("shared block table does not match prefix length")
        cache = cls(allocator, max_tokens=max_tokens)
        allocator.retain(blocks)
        cache.blocks = list(blocks)
        cache.length = length
        return cache

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def capacity(self) -> int:
        return self.max_tokens

    @property
    def allocated_tokens(self) -> int:
        return len(self.blocks) * self.allocator.block_size

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("paged cache is closed")

    def _ensure_blocks(self, end: int) -> None:
        required = self.allocator.blocks_for_tokens(end)
        missing = required - len(self.blocks)
        if missing > 0:
            self.blocks.extend(self.allocator.allocate(missing))

    def append(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        if keys.shape != values.shape or keys.ndim != 4 or keys.size(0) != 1:
            raise ValueError("states must have shape [1, tokens, kv_heads, head_dim]")
        if start != self.length:
            raise ValueError(f"cache cursor is {self.length}, received start={start}")
        end = start + keys.size(1)
        if end > self.max_tokens:
            raise ValueError(
                f"paged cache capacity exceeded: need {end}, have {self.max_tokens}"
            )
        self._ensure_blocks(end)

        source_offset = 0
        logical = start
        while logical < end:
            table_index, block_offset = divmod(logical, self.allocator.block_size)
            count = min(end - logical, self.allocator.block_size - block_offset)
            self.allocator.write(
                layer=layer,
                block=self.blocks[table_index],
                offset=block_offset,
                keys=keys[:, source_offset : source_offset + count],
                values=values[:, source_offset : source_offset + count],
            )
            logical += count
            source_offset += count
        return self.allocator.read(layer=layer, blocks=self.blocks, length=end)

    def advance(self, tokens: int) -> None:
        self._ensure_open()
        if tokens < 0 or self.length + tokens > self.max_tokens:
            raise ValueError("invalid paged-cache advance")
        self.length += tokens

    def truncate(self, length: int) -> None:
        self._ensure_open()
        if length < 0 or length > self.length:
            raise ValueError("cache can only truncate its active prefix")
        keep = self.allocator.blocks_for_tokens(length)
        released = self.blocks[keep:]
        self.blocks = self.blocks[:keep]
        if released:
            self.allocator.release(released)
        self.length = length

    def close(self) -> None:
        if not self._closed:
            self.allocator.release(self.blocks)
            self.blocks = []
            self.length = 0
            self._closed = True
