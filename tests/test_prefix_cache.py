import torch

from microserve import (
    BlockAllocator,
    ContinuousBatcher,
    ModelConfig,
    PagedKVCache,
    PrefixCache,
    Request,
    Transformer,
    generate_naive,
)


def setup() -> tuple[Transformer, BlockAllocator, PrefixCache]:
    torch.manual_seed(17)
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
        num_layers=2,
        num_blocks=32,
        block_size=2,
        num_kv_heads=1,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return model, pool, PrefixCache(pool)


def test_prefix_hit_reuses_blocks_and_preserves_generation() -> None:
    model, pool, prefixes = setup()
    batcher = ContinuousBatcher(
        model,
        cache_factory=lambda request: PagedKVCache(
            pool,
            max_tokens=request.prompt.numel() + request.max_new_tokens,
        ),
        prefix_cache=prefixes,
    )
    first = Request("first", torch.tensor([1, 2, 3, 4, 5]), 2)
    batcher.submit(first)
    while "first" not in batcher.results:
        batcher.step()

    second = Request("second", torch.tensor([1, 2, 3, 4, 9, 10]), 3)
    batcher.submit(second)

    state = next(state for state in batcher.prefilling if state.request is second)
    assert state.prompt_offset == 4
    assert prefixes.hits == 1
    assert all(pool.references(block) >= 2 for block in state.cache.blocks)

    while "second" not in batcher.results:
        batcher.step()
    expected = generate_naive(model, second.prompt[None], max_new_tokens=3)
    assert batcher.results["second"].generated == tuple(expected[0, 6:].tolist())

    prefixes.close()
    assert pool.used_blocks == 0


def test_exact_block_aligned_prompt_recomputes_one_block() -> None:
    model, pool, prefixes = setup()
    cache = PagedKVCache(pool, max_tokens=8)
    prompt = torch.tensor([1, 2, 3, 4])
    model(prompt[None], cache=cache)
    prefixes.insert(prompt, cache)
    cache.close()

    match = prefixes.lookup(prompt, max_tokens=8)

    assert match is not None
    shared, length = match
    assert length == 2
    shared.close()
    prefixes.close()
    assert pool.used_blocks == 0


def test_lru_eviction_releases_prefix_references() -> None:
    model, pool, _ = setup()
    prefixes = PrefixCache(pool, max_entries=1)
    for prompt in (torch.tensor([1, 2]), torch.tensor([3, 4])):
        cache = PagedKVCache(pool, max_tokens=4)
        model(prompt[None], cache=cache)
        prefixes.insert(prompt, cache)
        cache.close()

    assert len(prefixes) == 1
    assert pool.used_blocks == 1
    prefixes.close()
    assert pool.used_blocks == 0
