import torch

from microserve import (
    ContinuousBatcher,
    FCFSScheduler,
    ModelConfig,
    Request,
    SLOAwareScheduler,
    Transformer,
    generate_naive,
)


def tiny_model() -> Transformer:
    torch.manual_seed(11)
    return Transformer(
        ModelConfig(
            vocab_size=32,
            dim=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=32,
        )
    ).eval()


def test_ragged_decode_matches_independent_generation() -> None:
    model = tiny_model()
    requests = [
        Request("short", torch.tensor([1, 2]), 4),
        Request("long", torch.tensor([3, 4, 5, 6, 7]), 3),
    ]
    batcher = ContinuousBatcher(model)

    results = batcher.run(requests)

    for request in requests:
        expected = generate_naive(
            model, request.prompt[None, :], max_new_tokens=request.max_new_tokens
        )[0, request.prompt.numel() :]
        assert results[request.request_id].generated == tuple(expected.tolist())


def test_continuous_batching_admits_later_arrival() -> None:
    model = tiny_model()
    requests = [
        Request("first", torch.tensor([1, 2]), 4, arrival_step=0),
        Request("later", torch.tensor([3, 4]), 2, arrival_step=2),
    ]
    batcher = ContinuousBatcher(model)

    results = batcher.run(requests)

    assert results["first"].first_token_step == 1
    assert results["later"].first_token_step >= 3
    assert results["later"].completion_step <= results["first"].completion_step


def test_chunked_prefill_bounds_work_per_step() -> None:
    model = tiny_model()
    scheduler = FCFSScheduler(
        max_batch_size=4, max_batch_tokens=3, prefill_chunk_size=2
    )
    batcher = ContinuousBatcher(model, scheduler=scheduler)
    batcher.submit(Request("long", torch.tensor([1, 2, 3, 4, 5]), 1))

    steps = [batcher.step(), batcher.step(), batcher.step()]

    assert [step.scheduled.tokens for step in steps] == [2, 2, 1]
    assert batcher.results["long"].first_token_step == 3


class FakeState:
    def __init__(self, arrival: int, ttft: int) -> None:
        self.arrival_step = arrival
        self.prompt_remaining = 1
        self.last_token_step = None
        self.ttft_slo_steps = ttft
        self.itl_slo_steps = None


def test_slo_scheduler_prefers_earliest_deadline() -> None:
    scheduler = SLOAwareScheduler(max_batch_size=1, max_batch_tokens=1)
    relaxed = FakeState(arrival=0, ttft=10)
    urgent = FakeState(arrival=2, ttft=2)

    schedule = scheduler.schedule(prefilling=[relaxed, urgent], running=[], now=2)

    assert schedule.prefill[0].state is urgent
