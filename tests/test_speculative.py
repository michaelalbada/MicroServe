import math

import torch

from microserve import (
    ModelConfig,
    Transformer,
    generate_naive,
    generate_speculative,
)


def model(seed: int) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(
        ModelConfig(
            vocab_size=24,
            dim=24,
            num_layers=2,
            num_heads=3,
            num_kv_heads=1,
            max_seq_len=32,
        )
    ).eval()


def test_identical_draft_accepts_every_token_in_fewer_target_calls() -> None:
    target = model(19)
    prompt = torch.tensor([[1, 2, 3]])

    result = generate_speculative(
        target, target, prompt, max_new_tokens=7, draft_tokens=3
    )
    expected = generate_naive(target, prompt, max_new_tokens=7)

    torch.testing.assert_close(result.tokens, expected)
    assert result.acceptance_rate == 1.0
    assert result.target_calls == 1 + math.ceil(7 / 3)


def test_rejected_draft_rolls_back_and_matches_target() -> None:
    target = model(23)
    draft = model(29)
    with torch.no_grad():
        for parameter in draft.parameters():
            parameter.zero_()

    prompt_token = next(
        token
        for token in range(1, target.config.vocab_size)
        if int(target(torch.tensor([[token]]))[0, -1].argmax()) != 0
    )
    prompt = torch.tensor([[prompt_token]])
    result = generate_speculative(
        target, draft, prompt, max_new_tokens=6, draft_tokens=4
    )
    expected = generate_naive(target, prompt, max_new_tokens=6)

    torch.testing.assert_close(result.tokens, expected)
    assert result.accepted_draft_tokens < result.proposed_draft_tokens
    assert result.target_calls > 1


def test_zero_tokens_does_not_run_either_model() -> None:
    target = model(31)
    prompt = torch.tensor([[1, 2]])

    result = generate_speculative(target, target, prompt, max_new_tokens=0)

    assert result.tokens is prompt
    assert result.target_calls == result.draft_calls == 0
