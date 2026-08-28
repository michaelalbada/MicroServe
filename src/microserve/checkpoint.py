"""Fetch and load real Llama-format checkpoints into microServe's model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from safetensors import safe_open
from tokenizers import Tokenizer

from microserve.model import ModelConfig, Transformer


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DOWNLOAD_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class CheckpointConfig:
    model: ModelConfig
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    torch_dtype: str | None
    tie_word_embeddings: bool


class TextTokenizer:
    """The small text/token contract needed by the generation CLI."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        bos_token_id: int | None,
        eos_token_id: int | None,
        add_bos_token: bool,
    ) -> None:
        self._tokenizer = tokenizer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.add_bos_token = add_bos_token

    def encode(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=True).ids
        if self.add_bos_token and self.bos_token_id is not None:
            if not ids or ids[0] != self.bos_token_id:
                ids.insert(0, self.bos_token_id)
        return ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: Transformer
    tokenizer: TextTokenizer | None
    config: CheckpointConfig
    source: Path
    model_id: str


def fetch_snapshot(
    model_id: str = DEFAULT_MODEL,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Fetch only the files required for inference into the Hub cache."""
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=list(DOWNLOAD_PATTERNS),
        )
    )


def read_checkpoint_config(snapshot: str | Path) -> CheckpointConfig:
    snapshot = Path(snapshot)
    with (snapshot / "config.json").open() as file:
        raw: dict[str, Any] = json.load(file)

    if raw.get("model_type") != "llama":
        raise ValueError(
            f"microServe currently loads Llama checkpoints, got "
            f"model_type={raw.get('model_type')!r}"
        )
    if raw.get("attention_bias", False) or raw.get("mlp_bias", False):
        raise ValueError("attention/MLP bias weights are not supported")
    if raw.get("hidden_act", "silu") != "silu":
        raise ValueError("only the Llama SwiGLU/silu feed-forward is supported")
    if raw.get("rope_scaling") not in (None, {"type": "none"}):
        raise ValueError("RoPE scaling checkpoints are not yet supported")

    config = ModelConfig(
        vocab_size=int(raw["vocab_size"]),
        dim=int(raw["hidden_size"]),
        num_layers=int(raw["num_hidden_layers"]),
        num_heads=int(raw["num_attention_heads"]),
        num_kv_heads=int(raw.get("num_key_value_heads", raw["num_attention_heads"])),
        hidden_dim=int(raw["intermediate_size"]),
        max_seq_len=int(raw["max_position_embeddings"]),
        rope_base=float(raw.get("rope_theta", 10_000.0)),
        norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
    )
    return CheckpointConfig(
        model=config,
        bos_token_id=raw.get("bos_token_id"),
        eos_token_id=raw.get("eos_token_id"),
        pad_token_id=raw.get("pad_token_id"),
        torch_dtype=raw.get("torch_dtype"),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
    )


def resolve_dtype(
    requested: str, device: torch.device, checkpoint_dtype: str | None = None
) -> torch.dtype:
    if requested != "auto":
        choices = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        try:
            return choices[requested]
        except KeyError as error:
            raise ValueError(f"unknown dtype: {requested}") from error
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if checkpoint_dtype == "float32":
        return torch.float32
    return torch.float32


def _weight_name_map(config: ModelConfig) -> dict[str, str]:
    names = {
        "model.embed_tokens.weight": "token_embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for layer in range(config.num_layers):
        hf = f"model.layers.{layer}"
        micro = f"layers.{layer}"
        names.update(
            {
                f"{hf}.input_layernorm.weight": f"{micro}.attention_norm.weight",
                f"{hf}.self_attn.q_proj.weight": f"{micro}.attention.q_proj.weight",
                f"{hf}.self_attn.k_proj.weight": f"{micro}.attention.k_proj.weight",
                f"{hf}.self_attn.v_proj.weight": f"{micro}.attention.v_proj.weight",
                f"{hf}.self_attn.o_proj.weight": f"{micro}.attention.out_proj.weight",
                f"{hf}.post_attention_layernorm.weight": f"{micro}.ffn_norm.weight",
                f"{hf}.mlp.gate_proj.weight": f"{micro}.feed_forward.gate_proj.weight",
                f"{hf}.mlp.up_proj.weight": f"{micro}.feed_forward.up_proj.weight",
                f"{hf}.mlp.down_proj.weight": f"{micro}.feed_forward.down_proj.weight",
            }
        )
    return names


def _weight_files(snapshot: Path) -> list[Path]:
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        with index.open() as file:
            weight_map = json.load(file)["weight_map"]
        return [snapshot / name for name in sorted(set(weight_map.values()))]
    files = sorted(snapshot.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights found under {snapshot}")
    return files


def load_llama_weights(
    snapshot: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: str = "auto",
) -> tuple[Transformer, CheckpointConfig]:
    """Load HF Llama weights without constructing a Transformers model."""
    snapshot = Path(snapshot)
    device = torch.device(device)
    checkpoint = read_checkpoint_config(snapshot)
    names = _weight_name_map(checkpoint.model)
    loaded: dict[str, torch.Tensor] = {}

    for weight_file in _weight_files(snapshot):
        with safe_open(weight_file, framework="pt", device="cpu") as tensors:
            for hf_name in tensors.keys():
                micro_name = names.get(hf_name)
                if micro_name is not None:
                    loaded[micro_name] = tensors.get_tensor(hf_name)

    if checkpoint.tie_word_embeddings and "lm_head.weight" not in loaded:
        loaded["lm_head.weight"] = loaded["token_embedding.weight"]

    with torch.device("meta"):
        model = Transformer(checkpoint.model)
    expected = set(model.state_dict())
    missing = sorted(expected - loaded.keys())
    unexpected = sorted(loaded.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint mapping mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(loaded, strict=True, assign=True)
    loaded.clear()
    if checkpoint.tie_word_embeddings:
        model.lm_head.weight = model.token_embedding.weight
    target_dtype = resolve_dtype(dtype, device, checkpoint.torch_dtype)
    model = model.to(device=device, dtype=target_dtype).eval()
    return model, checkpoint


def load_tokenizer(snapshot: str | Path, config: CheckpointConfig) -> TextTokenizer:
    snapshot = Path(snapshot)
    tokenizer_config_path = snapshot / "tokenizer_config.json"
    tokenizer_config: dict[str, Any] = {}
    if tokenizer_config_path.exists():
        with tokenizer_config_path.open() as file:
            tokenizer_config = json.load(file)
    return TextTokenizer(
        Tokenizer.from_file(str(snapshot / "tokenizer.json")),
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        add_bos_token=bool(tokenizer_config.get("add_bos_token", False)),
    )


def load_checkpoint(
    model_id: str = DEFAULT_MODEL,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    device: torch.device | str = "cpu",
    dtype: str = "auto",
    with_tokenizer: bool = True,
) -> LoadedCheckpoint:
    snapshot = fetch_snapshot(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model, config = load_llama_weights(snapshot, device=device, dtype=dtype)
    tokenizer = load_tokenizer(snapshot, config) if with_tokenizer else None
    return LoadedCheckpoint(model, tokenizer, config, snapshot, model_id)


def _snapshot_size(snapshot: Path) -> int:
    seen = set()
    total = 0
    for path in snapshot.iterdir():
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            total += resolved.stat().st_size
    return total


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a supported real model.")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold]Fetching[/bold] {args.model}")
    try:
        snapshot = fetch_snapshot(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )
        config = read_checkpoint_config(snapshot)
    except Exception as error:
        console.print(Panel(str(error), title="Model fetch failed", border_style="red"))
        raise SystemExit(2) from error

    console.print(
        Panel.fit(
            f"[bold green]Ready[/bold green]  {args.model}\n"
            f"Path: {snapshot}\n"
            f"Download size: {_format_bytes(_snapshot_size(snapshot))}\n"
            f"Parameters config: {config.model.num_layers} layers, "
            f"width {config.model.dim}, vocab {config.model.vocab_size}",
            title="microServe model cache",
        )
    )


if __name__ == "__main__":
    main()
