# microServe

Modern LLM serving from first principles.

Sampling one token is easy. The serving problem is keeping expensive
accelerators useful while many autoregressive requests arrive, grow, finish,
share prefixes, and compete for KV memory under latency objectives.

`microServe` builds that system in nine executable stages. Stages 0–6 run one
tiny PyTorch decoder and verify token-for-token equivalence. Stages 7–8 use a
deterministic cost simulator so distributed queues and KV transport are visible
without pretending that local function calls are a network deployment.

No vLLM, SGLang, TensorRT-LLM, or serving framework is used.

## Run the curriculum

Install [uv](https://docs.astral.sh/uv/), then create the locked development
environment and run the curriculum:

```bash
uv sync
uv run pytest
uv run microserve quickstart
```

`microserve quickstart` is network-free and takes only a few seconds. It sweeps
three prompt lengths through naive and KV-cached generation, verifies identical
tokens, and shows how the cache advantage grows with reusable context.

The unified command surface is:

```text
uv run microserve quickstart   # fast first experiment; no download
uv run microserve generate     # fetch/use a real model and complete text
uv run microserve bench        # controlled naive/KV timing experiments
uv run microserve scorecard    # execute serving configurations with a real model
uv run microserve fetch        # explicitly populate the model cache
```

The scorecard loads the default real model, synchronizes the selected device,
and runs the same request workload through five executable serving designs:

```text
Measured serving configurations — synchronized wall time
configuration  cache       policy  correct  wall  p50 TTFT  p95 TTFT  p95 ITL  tok/s  blocks
sequential     contiguous  FCFS       True   ...       ...       ...      ...    ...       -
continuous     contiguous  FCFS       True   ...       ...       ...      ...    ...       -
paged          paged       FCFS       True   ...       ...       ...      ...    ...     ...
chunked        paged       FCFS       True   ...       ...       ...      ...    ...     ...
slo_aware      paged       SLO        True   ...       ...       ...      ...    ...     ...
```

Every row performs model forward passes and records token emission timestamps.
TTFT, ITL, wall time, throughput, correctness, and paged-block usage are all
observed; there is no fitted service rate, virtual replica, or assumed network.
`--prompt-lengths` creates one concurrent request per length, and
`--decode-steps` is the output length of each request. Use `--configurations`
to select a smaller subset for long experiments.

For a focused stage-0/1 timing run:

```bash
uv run microserve bench --batch-size 4 --prompt-tokens 128 --new-tokens 32
```

That default is deliberately a randomly initialized 5.8M-parameter teaching
model. The CLI labels it `random toy weights (no model download)` so mechanism
timings cannot be mistaken for useful-model inference.

## Run a real model

The built-in small default is
[`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M):
a 134.5M-parameter Llama model whose required files occupy about 259 MiB.

```bash
# Fetch weights and tokenizer into the standard Hugging Face cache.
uv run microserve fetch HuggingFaceTB/SmolLM2-135M

# Generate text through microServe's own model and KV cache.
uv run microserve generate "The capital of France is" --device mps --new-tokens 32

# Execute the serving configuration sweep with this model/device.
uv run microserve scorecard --model HuggingFaceTB/SmolLM2-135M --device mps

# Benchmark those same real weights. Skip naive recomputation for larger runs.
uv run microserve bench \
  --model HuggingFaceTB/SmolLM2-135M \
  --prompt "The capital of France is" \
  --methods kv_cache \
  --batch-size 4 \
  --new-tokens 32
```

The Hub client supplies download/cache management, `tokenizers` supplies text
encoding, and `safetensors` supplies safe tensor reads. The actual model forward
pass, RoPE, GQA, KV cache, and generation loop remain this repository's PyTorch
implementation; no Transformers model is constructed.

The loader currently accepts standard Llama safetensors, including sharded
checkpoints, with SwiGLU, bias-free attention/MLP, ordinary RoPE, and optional
tied embeddings. Unsupported architectures or RoPE scaling fail explicitly.

Use `--cache-dir`, `--revision`, and `--offline` to control model resolution.
`--dtype auto` selects FP16 on MPS, BF16/FP16 on CUDA, and FP32 on CPU.

## CLI memory and progress

Before constructing a model or allocating KV state, benchmark, scorecard, and
generation commands show:

- model provenance, parameter count, device, and dtype;
- model, KV-cache, attention, activation, and MPS allocator-churn estimates;
- estimated peak versus a configurable safe fraction of available memory;
- a refusal with concrete smaller-run suggestions when the estimate is unsafe.

The benchmark can isolate a mechanism with `--methods naive` or
`--methods kv_cache`. `--force` bypasses preflight but does not make an unsafe
allocation safe. Backend OOMs are caught, unused device memory is released, and
the raw traceback is replaced with an actionable error panel.

Long generation loops show token progress and estimated remaining time. Use
`--no-progress` when collecting timings where even small UI overhead matters.

## The nine stages

### 0. Naive generation

The reference recomputes the full sequence for every token:

```python
for _ in range(max_new_tokens):
    token = model(tokens)[:, -1].argmax(dim=-1, keepdim=True)
    tokens = torch.cat((tokens, token), dim=1)
```

Every later model-executing optimization must produce the same tokens.

### 1. KV caching

Prefill the prompt once, then append one key and value per layer per decode
token. `KVCache` preallocates contiguous storage and exposes its single shared
cursor—the limitation that motivates per-request state and paging.

```python
logits = model(prompt, cache=cache)       # prefill
logits = model(next_token, cache=cache)   # decode
```

### 2. Continuous batching

`ContinuousBatcher` admits new requests, prefills waiting requests, performs one
ragged decode batch over independently sized caches, and retires completed
requests on every iteration.

```python
while pending or prefilling or running:
    schedule = scheduler.schedule(prefilling, running)
    model.decode_batch(tokens, caches=request_caches)
```

Requests retain independent token histories and cache lengths; attention pads
only the physical decode batch. Results record TTFT and every inter-token
interval in deterministic engine steps.

### 3. Paged KV allocation

`BlockAllocator` owns fixed-size physical K/V blocks. Each `PagedKVCache` maps a
request's logical positions through its block table, so growth no longer needs a
maximum-length contiguous reservation. `paged_attention()` consumes that block
table directly: it visits physical pages in logical order and combines them
with an online softmax. It never concatenates a request's KV or pads the batch
to its longest sequence.

```python
for logical_block, physical_block in enumerate(block_table):
    keys = physical_keys[physical_block]
    values = physical_values[physical_block]
    scores = query @ keys.transpose(-1, -2)
    output = online_softmax_update(output, scores, values)
```

This is a portable PyTorch reference kernel rather than a fused accelerator
kernel. Its purpose is to make the PagedAttention memory access and numerics
real and inspectable on CPU, CUDA, and MPS; an optimized kernel can implement
the same `PagedKVView` contract later.

```text
request logical blocks:  [0] [1] [2]
                           |   |   |
physical block IDs:       11   3  27
```

Blocks are ref-counted, allocation failure is explicit, speculative rollback
releases unused tail blocks, and request completion returns storage to the pool.

### 4. Prefix caching

`PrefixCache` is a small LRU over block-aligned token prefixes. Full physical
blocks are immutable and shared by reference. A hit reuses the longest matching
prefix while recomputing at least one prompt token to recover next-token logits.

This makes the important distinction explicit: cached KV state is not itself a
cached next-token prediction.

### 5. Chunked prefill

Long prompts consume the scheduler's token budget in bounded chunks. Decode is
scheduled first, then remaining capacity is assigned to prefill:

```python
budget -= len(decode_batch)
chunk = min(prompt_remaining, prefill_chunk_size, budget)
```

This can reduce decode interference while increasing the long prompt's own
TTFT. The scorecard reports that tradeoff instead of declaring one chunk size
universally optimal.

### 6. Speculative decoding

The draft model proposes several tokens. One target call verifies the group.
Matching tokens are accepted; at the first mismatch both caches roll back to
the accepted prefix and append the target correction.

```python
proposals = draft.generate(k)
target_predictions = target.verify(proposals)
accepted, correction = first_mismatch(proposals, target_predictions)
target_cache.truncate(prefix + len(accepted))
```

The implementation is greedy and single-request to keep the acceptance and
rollback rules readable. Tests cover perfect and deliberately bad drafts.

### Prefill/decode disaggregation

`DisaggregatedSimulator` separates prefill workers, a serialized KV-transfer
link, and decode workers. Every result decomposes TTFT into:

```text
prefill queue + prefill compute
              + transfer queue + KV transfer
                               + decode queue + first decode token
```

Prefill and decode pools can have different sizes and token rates. The simulator
is deterministic, making queueing, bandwidth, and fixed-latency assumptions
easy to change and test.

### SLO-aware scheduling and routing

`SLOAwareScheduler` orders live prefill and decode work by remaining TTFT or ITL
slack. `SLOAwareRouter` predicts completion using queue availability, worker
rate, fixed cost, KV-transfer cost, and cached-prefix locality. The baseline
router sees only queue availability, demonstrating why "shortest queue" can
choose the slower path.

The scorecard executes `SLOAwareScheduler` directly and reports observed TTFT,
ITL, output throughput, and correctness. The disaggregation simulator remains
an inspectable teaching component, but its modeled times are not presented as
device measurements.

## Repository map

```text
src/microserve/
    model.py          tiny decoder: RoPE, GQA, RMSNorm, SwiGLU
    generate.py       stage 0 and 1 generation loops
    kv_cache.py       contiguous cache and shared cache protocol
    batcher.py        request lifecycle and ragged continuous decode
    scheduler.py      FCFS/chunked and earliest-slack-first policies
    allocator.py      physical block pool and paged request caches
    paged_attention.py indirect block-table attention with online softmax
    profiler.py       synchronized real-model prefill/decode measurements
    prefix_cache.py   block-aligned prefix LRU
    speculative.py    draft, verify, accept, and rollback
    disagg.py         prefill/transfer/decode cost simulator
    router.py         queue-only and predictive SLO routing
    checkpoint.py     Hub download and Llama safetensors weight mapping
    memory.py         preflight estimates and device cleanup
    cli_ui.py         shared Rich panels, tables, and error presentation
    cli.py            unified microserve subcommand dispatcher
    quickstart.py     network-free KV-cache crossover experiment
    infer.py          real-model text generation CLI
    benchmark.py      focused wall-clock benchmark
    scorecard.py      real-model serving configuration measurements
tests/
    test_*.py         mechanism and configuration-sweep tests
```

## Scope

This is executable teaching code, not a production server. It deliberately
omits HTTP/RPC, tokenizer plumbing, CUDA kernels, tensor parallelism, fault
tolerance, and multi-node process management. Those concerns matter, but adding
them here would obscure the memory and scheduling mechanisms this project is
meant to explain.

## Shared contract with microTrain

`microTrain` should export the same `ModelConfig` fields and a standard PyTorch
`state_dict`. `microServe` owns inference-time state and policy; it does not
invent a serving-only checkpoint format.

```text
microTrain checkpoint -> microServe workload -> quality + serving metrics
```
