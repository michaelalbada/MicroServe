"""A tiny decoder model shared by every serving stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from microserve.kv_cache import Cache, KVCache


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 2
    hidden_dim: int | None = None
    max_seq_len: int = 2048
    rope_base: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (self.dim // self.num_heads) % 2:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def mlp_dim(self) -> int:
        return self.hidden_dim or 4 * self.dim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale).to(x.dtype) * self.weight


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Apply rotary position embeddings to [batch, tokens, heads, head_dim]."""
    half_dim = x.size(-1) // 2
    inv_freq = base ** (
        -torch.arange(half_dim, device=x.device, dtype=torch.float32) / half_dim
    )
    angles = positions.float()[..., None] * inv_freq
    if positions.ndim == 1:
        cos = angles.cos().to(x.dtype)[None, :, None, :]
        sin = angles.sin().to(x.dtype)[None, :, None, :]
    elif positions.ndim == 2:
        cos = angles.cos().to(x.dtype)[:, :, None, :]
        sin = angles.sin().to(x.dtype)[:, :, None, :]
    else:
        raise ValueError("positions must have shape [tokens] or [batch, tokens]")
    left, right = x.chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.q_proj = nn.Linear(
            config.dim, config.num_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.dim, config.num_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.dim, config.num_kv_heads * config.head_dim, bias=False
        )
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(
        self, x: torch.Tensor, *, start_pos: int, cache: Cache | None
    ) -> torch.Tensor:
        batch, tokens, _ = x.shape
        config = self.config
        positions = torch.arange(
            start_pos, start_pos + tokens, device=x.device, dtype=torch.long
        )

        q = self.q_proj(x).view(batch, tokens, config.num_heads, config.head_dim)
        k = self.k_proj(x).view(batch, tokens, config.num_kv_heads, config.head_dim)
        v = self.v_proj(x).view(batch, tokens, config.num_kv_heads, config.head_dim)
        q = _apply_rope(q, positions, config.rope_base)
        k = _apply_rope(k, positions, config.rope_base)

        if cache is not None:
            k, v = cache.append(self.layer_id, k, v, start=start_pos)

        # Expand grouped KV heads explicitly so the data movement is visible.
        # A later kernel-focused stage can use native GQA without changing policy.
        groups = config.num_heads // config.num_kv_heads
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))

        key_positions = torch.arange(k.size(2), device=x.device)
        causal = key_positions[None, :] <= positions[:, None]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, config.dim)
        return self.out_proj(attended)

    def decode_batch(self, x: torch.Tensor, caches: list[Cache]) -> torch.Tensor:
        """Decode one token for requests whose cache lengths may differ."""
        batch, tokens, _ = x.shape
        if tokens != 1 or len(caches) != batch:
            raise ValueError("ragged decode expects one token and one cache per row")

        config = self.config
        positions = torch.tensor(
            [cache.length for cache in caches], device=x.device, dtype=torch.long
        )[:, None]
        q = self.q_proj(x).view(batch, 1, config.num_heads, config.head_dim)
        k = self.k_proj(x).view(batch, 1, config.num_kv_heads, config.head_dim)
        v = self.v_proj(x).view(batch, 1, config.num_kv_heads, config.head_dim)
        q = _apply_rope(q, positions, config.rope_base)
        k = _apply_rope(k, positions, config.rope_base)

        prefixes = [
            cache.append(
                self.layer_id,
                k[row : row + 1],
                v[row : row + 1],
                start=cache.length,
            )
            for row, cache in enumerate(caches)
        ]
        lengths = [keys.size(1) for keys, _ in prefixes]
        max_length = max(lengths)
        padded_k = x.new_zeros(batch, max_length, config.num_kv_heads, config.head_dim)
        padded_v = torch.zeros_like(padded_k)
        for row, (keys, values) in enumerate(prefixes):
            padded_k[row, : lengths[row]] = keys[0]
            padded_v[row, : lengths[row]] = values[0]

        groups = config.num_heads // config.num_kv_heads
        padded_k = padded_k.repeat_interleave(groups, dim=2)
        padded_v = padded_v.repeat_interleave(groups, dim=2)
        q = q.transpose(1, 2)
        padded_k = padded_k.transpose(1, 2)
        padded_v = padded_v.transpose(1, 2)
        valid = (
            torch.arange(max_length, device=x.device)[None, :]
            < torch.tensor(lengths, device=x.device)[:, None]
        )
        attended = F.scaled_dot_product_attention(
            q, padded_k, padded_v, attn_mask=valid[:, None, None, :]
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, 1, config.dim)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.mlp_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.mlp_dim, bias=False)
        self.down_proj = nn.Linear(config.mlp_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = Attention(config, layer_id)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(
        self, x: torch.Tensor, *, start_pos: int, cache: Cache | None
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), start_pos=start_pos, cache=cache)
        return x + self.feed_forward(self.ffn_norm(x))

    def decode_batch(self, x: torch.Tensor, caches: list[Cache]) -> torch.Tensor:
        x = x + self.attention.decode_batch(self.attention_norm(x), caches)
        return x + self.feed_forward(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_id) for layer_id in range(config.num_layers)
        )
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(
        self, *, batch_size: int, max_tokens: int, dtype: torch.dtype | None = None
    ) -> KVCache:
        parameter = next(self.parameters())
        return KVCache(
            num_layers=self.config.num_layers,
            batch_size=batch_size,
            max_tokens=max_tokens,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            device=parameter.device,
            dtype=dtype or parameter.dtype,
        )

    def forward(
        self, token_ids: torch.Tensor, *, cache: Cache | None = None
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if token_ids.size(1) == 0:
            raise ValueError("at least one token is required")
        if cache is not None and token_ids.size(0) != cache.batch_size:
            raise ValueError("token batch size does not match KV cache")

        start_pos = cache.length if cache is not None else 0
        end_pos = start_pos + token_ids.size(1)
        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {end_pos} exceeds model limit "
                f"{self.config.max_seq_len}"
            )

        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x, start_pos=start_pos, cache=cache)
        if cache is not None:
            cache.advance(token_ids.size(1))
        return self.lm_head(self.norm(x))

    def decode_batch(
        self, token_ids: torch.Tensor, *, caches: list[Cache]
    ) -> torch.Tensor:
        """Run one physical decode batch over independently growing requests."""
        if token_ids.ndim != 2 or token_ids.size(1) != 1:
            raise ValueError("decode_batch expects token_ids with shape [batch, 1]")
        if len(caches) != token_ids.size(0):
            raise ValueError("one cache is required for each token row")
        if any(cache.batch_size != 1 for cache in caches):
            raise ValueError("ragged decode caches must each contain one request")
        if any(cache.length >= self.config.max_seq_len for cache in caches):
            raise ValueError("a request exceeds the model sequence limit")

        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer.decode_batch(x, caches)
        for cache in caches:
            cache.advance(1)
        return self.lm_head(self.norm(x))
