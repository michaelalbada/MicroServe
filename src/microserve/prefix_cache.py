"""Stage 4: reuse block-aligned KV prefixes across requests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from microserve.allocator import BlockAllocator, PagedKVCache


@dataclass(frozen=True)
class PrefixEntry:
    tokens: tuple[int, ...]
    blocks: tuple[int, ...]


class PrefixCache:
    """A small LRU of immutable, full KV blocks indexed by token prefixes.

    Only complete blocks are shared, so a request always appends into a block it
    owns. The final prompt token is deliberately recomputed because KV state
    alone does not contain the logits needed to sample its successor.
    """

    def __init__(self, allocator: BlockAllocator, *, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.allocator = allocator
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], PrefixEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0

    def insert(self, prompt: torch.Tensor, cache: PagedKVCache) -> None:
        if cache.allocator is not self.allocator:
            raise ValueError("prefix cache and request cache use different allocators")
        tokens = tuple(int(token) for token in prompt.tolist())
        block_size = self.allocator.block_size
        complete = min(len(tokens), cache.length) // block_size

        for block_count in range(1, complete + 1):
            length = block_count * block_size
            key = tokens[:length]
            if key in self._entries:
                self._entries.move_to_end(key)
                continue
            blocks = cache.blocks[:block_count]
            self.allocator.retain(blocks)
            self._entries[key] = PrefixEntry(key, tuple(blocks))
            self._evict_if_needed()

    def lookup(
        self, prompt: torch.Tensor, *, max_tokens: int
    ) -> tuple[PagedKVCache, int] | None:
        tokens = tuple(int(token) for token in prompt.tolist())
        block_size = self.allocator.block_size
        # Leave at least one prompt token to recompute the next-token logits.
        longest = ((len(tokens) - 1) // block_size) * block_size
        for length in range(longest, 0, -block_size):
            key = tokens[:length]
            entry = self._entries.get(key)
            if entry is None:
                continue
            self._entries.move_to_end(key)
            self.hits += 1
            self.reused_tokens += length
            return (
                PagedKVCache.from_shared_blocks(
                    self.allocator,
                    blocks=list(entry.blocks),
                    length=length,
                    max_tokens=max_tokens,
                ),
                length,
            )
        self.misses += 1
        return None

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            _, entry = self._entries.popitem(last=False)
            self.allocator.release(list(entry.blocks))

    def close(self) -> None:
        for entry in self._entries.values():
            self.allocator.release(list(entry.blocks))
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
