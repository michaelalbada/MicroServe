"""Stage 6: greedy speculative decoding with explicit verification/rollback."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from microserve.model import Transformer


@dataclass(frozen=True)
class SpeculativeResult:
    tokens: torch.Tensor
    target_calls: int
    draft_calls: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_draft_tokens


def _greedy(logits: torch.Tensor) -> int:
    return int(logits[0, -1].argmax().item())


@torch.inference_mode()
def generate_speculative(
    target: Transformer,
    draft: Transformer,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    draft_tokens: int = 4,
) -> SpeculativeResult:
    """Verify several cheap draft tokens in one target-model call.

    This is greedy speculation for one request, which keeps the acceptance rule
    visible: accept matching draft tokens, replace the first mismatch with the
    target token, and roll both caches back to the accepted prefix.
    """
    if prompt.ndim != 2 or prompt.size(0) != 1 or prompt.size(1) == 0:
        raise ValueError("speculative decoding expects one non-empty prompt")
    if max_new_tokens < 0 or draft_tokens < 1:
        raise ValueError("token counts must be non-negative and draft_tokens positive")
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("target and draft vocabularies must match")
    if max_new_tokens == 0:
        return SpeculativeResult(prompt, 0, 0, 0, 0)

    capacity = prompt.size(1) + max_new_tokens
    if capacity > min(target.config.max_seq_len, draft.config.max_seq_len):
        raise ValueError("generation exceeds a model sequence limit")
    target_cache = target.new_cache(batch_size=1, max_tokens=capacity)
    draft_cache = draft.new_cache(batch_size=1, max_tokens=capacity)

    output = prompt
    proposed_total = 0
    accepted_total = 0
    target_calls = 0
    draft_calls = 0
    try:
        target_logits = target(prompt, cache=target_cache)
        draft_logits = draft(prompt, cache=draft_cache)
        target_calls += 1
        draft_calls += 1

        generated = 0
        while generated < max_new_tokens:
            count = min(draft_tokens, max_new_tokens - generated)
            target_start = target_cache.length
            draft_start = draft_cache.length
            proposals = []
            for _ in range(count):
                token = _greedy(draft_logits)
                proposals.append(token)
                token_tensor = torch.tensor(
                    [[token]], device=prompt.device, dtype=torch.long
                )
                draft_logits = draft(token_tensor, cache=draft_cache)
                draft_calls += 1
            proposed_total += count

            proposal_tensor = torch.tensor(
                [proposals], device=prompt.device, dtype=torch.long
            )
            verified_logits = target(proposal_tensor, cache=target_cache)
            target_calls += 1
            target_predictions = [_greedy(target_logits)]
            target_predictions.extend(
                int(verified_logits[0, index].argmax().item())
                for index in range(count - 1)
            )

            mismatch = next(
                (
                    index
                    for index, (draft_token, target_token) in enumerate(
                        zip(proposals, target_predictions)
                    )
                    if draft_token != target_token
                ),
                None,
            )
            if mismatch is None:
                accepted = proposals
                accepted_total += len(accepted)
                output = torch.cat((output, proposal_tensor), dim=1)
                generated += len(accepted)
                target_logits = verified_logits
                continue

            accepted = proposals[:mismatch]
            correction = target_predictions[mismatch]
            accepted_total += len(accepted)
            emitted = torch.tensor(
                [accepted + [correction]], device=prompt.device, dtype=torch.long
            )
            output = torch.cat((output, emitted), dim=1)
            generated += emitted.size(1)

            # Verification and drafting ran beyond the accepted prefix. Rewind
            # both caches, then append the target's correction to re-align them.
            target_cache.truncate(target_start + mismatch)
            draft_cache.truncate(draft_start + mismatch)
            correction_tensor = torch.tensor(
                [[correction]], device=prompt.device, dtype=torch.long
            )
            target_logits = target(correction_tensor, cache=target_cache)
            draft_logits = draft(correction_tensor, cache=draft_cache)
            target_calls += 1
            draft_calls += 1
    finally:
        target_cache.close()
        draft_cache.close()

    return SpeculativeResult(
        output,
        target_calls,
        draft_calls,
        proposed_total,
        accepted_total,
    )
