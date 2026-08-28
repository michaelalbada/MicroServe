import json

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from microserve.checkpoint import (
    _weight_name_map,
    load_llama_weights,
    load_tokenizer,
    read_checkpoint_config,
)
from microserve.model import ModelConfig, Transformer


def write_config(path, config: ModelConfig) -> None:
    path.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "attention_bias": False,
                "mlp_bias": False,
                "hidden_act": "silu",
                "vocab_size": config.vocab_size,
                "hidden_size": config.dim,
                "num_hidden_layers": config.num_layers,
                "num_attention_heads": config.num_heads,
                "num_key_value_heads": config.num_kv_heads,
                "intermediate_size": config.mlp_dim,
                "max_position_embeddings": config.max_seq_len,
                "rope_theta": config.rope_base,
                "rms_norm_eps": config.norm_eps,
                "rope_scaling": None,
                "tie_word_embeddings": True,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 2,
                "torch_dtype": "float32",
            }
        )
    )


def synthetic_snapshot(tmp_path):
    config = ModelConfig(
        vocab_size=16,
        dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        hidden_dim=32,
        max_seq_len=32,
        rope_base=100_000,
    )
    torch.manual_seed(47)
    original = Transformer(config).eval()
    write_config(tmp_path / "config.json", config)
    micro_state = original.state_dict()
    hf_state = {
        hf_name: micro_state[micro_name].contiguous()
        for hf_name, micro_name in _weight_name_map(config).items()
        if hf_name != "lm_head.weight"
    }
    save_file(hf_state, tmp_path / "model.safetensors")
    return original, config


def test_load_real_format_weights_without_transformers(tmp_path) -> None:
    original, _ = synthetic_snapshot(tmp_path)

    loaded, checkpoint = load_llama_weights(tmp_path, device="cpu", dtype="float32")

    prompt = torch.tensor([[1, 4, 7]])
    torch.testing.assert_close(loaded(prompt), original(prompt))
    assert loaded.token_embedding.weight.data_ptr() == loaded.lm_head.weight.data_ptr()
    assert checkpoint.bos_token_id == 1


def test_load_tokenizer_from_downloaded_snapshot(tmp_path) -> None:
    _, config = synthetic_snapshot(tmp_path)
    tokenizer = Tokenizer(
        WordLevel({"<unk>": 0, "hello": 3, "world": 4}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"add_bos_token": True}))
    checkpoint = read_checkpoint_config(tmp_path)

    loaded = load_tokenizer(tmp_path, checkpoint)

    assert loaded.encode("hello world") == [1, 3, 4]
    assert loaded.decode([3, 4]) == "hello world"
    assert checkpoint.model == config


def test_rejects_non_llama_checkpoint(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))

    try:
        read_checkpoint_config(tmp_path)
    except ValueError as error:
        assert "currently loads Llama" in str(error)
    else:
        raise AssertionError("expected unsupported architecture to fail")
