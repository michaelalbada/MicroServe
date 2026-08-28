import pytest
import torch

from microserve import (
    BlockAllocator,
    ContinuousBatcher,
    ModelConfig,
    OutOfBlocksError,
    PagedKVCache,
    Request,
    Transformer,
    generate_naive,
)


def allocator(num_blocks: int = 4, block_size: int = 2) -> BlockAllocator:
    return BlockAllocator(
        num_layers=2,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_allocator_reuses_noncontiguous_physical_blocks() -> None:
    pool = allocator()
    first, middle, last = pool.allocate(3)
    pool.release([middle])

    reused = pool.allocate()[0]

    assert reused == middle
    assert {first, reused, last} == {0, 1, 2}
    assert pool.used_blocks == 3


def test_allocator_reference_counts_shared_blocks() -> None:
    pool = allocator()
    blocks = pool.allocate(2)
    pool.retain(blocks)
    pool.release(blocks)

    assert [pool.references(block) for block in blocks] == [1, 1]
    pool.release(blocks)
    assert pool.free_blocks == pool.num_blocks


def test_allocator_exhaustion_is_explicit_and_transactional() -> None:
    pool = allocator(num_blocks=2)
    pool.allocate()

    with pytest.raises(OutOfBlocksError, match="only 1"):
        pool.allocate(2)

    assert pool.free_blocks == 1


def test_paged_cache_matches_contiguous_generation() -> None:
    torch.manual_seed(13)
    model = Transformer(
        ModelConfig(
            vocab_size=32,
            dim=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=1,
            max_seq_len=32,
        )
    ).eval()
    pool = BlockAllocator(
        num_layers=model.config.num_layers,
        num_blocks=32,
        block_size=2,
        num_kv_heads=model.config.num_kv_heads,
        head_dim=model.config.head_dim,
        device=torch.device("cpu"),
        dtype=next(model.parameters()).dtype,
    )
    requests = [
        Request("a", torch.tensor([1, 2, 3]), 4),
        Request("b", torch.tensor([4, 5, 6, 7, 8]), 3),
    ]
    batcher = ContinuousBatcher(
        model,
        cache_factory=lambda request: PagedKVCache(
            pool,
            max_tokens=request.prompt.numel() + request.max_new_tokens,
        ),
    )

    results = batcher.run(requests)

    for request in requests:
        expected = generate_naive(
            model, request.prompt[None], max_new_tokens=request.max_new_tokens
        )[0, request.prompt.numel() :]
        assert results[request.request_id].generated == tuple(expected.tolist())
    assert pool.used_blocks == 0


def test_paged_cache_releases_blocks_after_truncation_and_close() -> None:
    pool = allocator(num_blocks=4, block_size=2)
    cache = PagedKVCache(pool, max_tokens=8)
    states = torch.zeros(1, 5, 1, 4)
    cache.append(0, states, states, start=0)
    cache.advance(5)
    assert pool.used_blocks == 3

    cache.truncate(2)
    assert pool.used_blocks == 1
    cache.close()
    assert pool.used_blocks == 0
