import torch

from microserve import ModelConfig, Transformer, generate_cached, generate_naive


def tiny_model() -> Transformer:
    torch.manual_seed(7)
    model = Transformer(
        ModelConfig(
            vocab_size=32,
            dim=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=32,
        )
    )
    return model.eval()


def test_cached_prefill_matches_regular_forward() -> None:
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    cache = model.new_cache(batch_size=2, max_tokens=8)

    expected = model(prompt)
    actual = model(prompt, cache=cache)

    torch.testing.assert_close(actual, expected)
    assert cache.length == prompt.size(1)


def test_cached_generation_matches_naive_generation() -> None:
    model = tiny_model()
    prompt = torch.tensor([[1, 5, 9], [2, 6, 10]])

    expected = generate_naive(model, prompt, max_new_tokens=5)
    actual = generate_cached(model, prompt, max_new_tokens=5)

    torch.testing.assert_close(actual, expected)


def test_zero_new_tokens_returns_prompt() -> None:
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3]])

    assert generate_naive(model, prompt, max_new_tokens=0) is prompt
    assert generate_cached(model, prompt, max_new_tokens=0) is prompt
