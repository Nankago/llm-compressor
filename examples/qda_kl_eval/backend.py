"""Transformers BF16 reference backend for local checkpoints."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def sha_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checkpoint_hashes(directory):
    directory = Path(directory)
    files = sorted(directory.glob("*.safetensors")) or sorted(
        directory.glob("pytorch_model*.bin")
    )
    if not files:
        raise ValueError("No reference weight shards found")
    return {p.name: sha_file(p) for p in files}


def tokenizer_info(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    backend = json.loads(tok.backend_tokenizer.to_str())
    # Padding/truncation settings of a previous encode are transient.
    for key in ("padding", "truncation"):
        backend.pop(key, None)
    payload = json.dumps([backend, tok.special_tokens_map], sort_keys=True).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "class": type(tok).__name__,
        "length": len(tok),
        "pad_id": tok.pad_token_id if tok.pad_token_id is not None else 0,
    }


def granite_packed(path, config):
    """Restore this example's linearized Granite weights with the upstream codec."""
    from compressed_tensors.compressors.pack_quantized import (
        PackedQuantizationCompressor,
    )
    from compressed_tensors.offload import remove_dispatch
    from compressed_tensors.quantization import QuantizationScheme
    from llmcompressor.modeling.moe.linearize import linearize_moe, repack_moe

    qc = config.quantization_config
    groups = list(qc["config_groups"].values())
    if len(groups) != 1 or groups[0]["format"] != "pack-quantized":
        raise ValueError("Expected one pack-quantized Granite recipe")
    scheme = QuantizationScheme.model_validate(groups[0])
    del config.quantization_config
    # Linearize on CPU, then restore native fused experts for inference.
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.bfloat16, attn_implementation="eager"
    )
    linearize_moe(model)
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as src:
        groups = {}
        for key in src.keys():
            name, suffix = key.rsplit(".", 1)
            groups.setdefault(name, []).append(suffix)
        dense = {}
        packed = []
        for name, suffixes in groups.items():
            state = {s: src.get_tensor(f"{name}.{s}") for s in suffixes}
            if "weight_packed" in state:
                if state["weight_packed"].dtype != torch.int32 or "weight" in state:
                    raise ValueError(f"Invalid packed weight at {name}")
                weight = PackedQuantizationCompressor.decompress(state, scheme)[
                    "weight"
                ]
                dense[name + ".weight"] = weight
                packed.append(name)
            else:
                dense.update({f"{name}.{s}": t for s, t in state.items()})
    expected = {
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and n not in qc["ignore"]
    }
    if set(packed) != expected or not packed:
        raise ValueError(f"Packed module coverage mismatch: {set(packed) ^ expected}")
    if config.tie_word_embeddings and "lm_head.weight" not in dense:
        dense["lm_head.weight"] = dense["model.embed_tokens.weight"]
    model.load_state_dict(dense, strict=True, assign=True)
    if config.tie_word_embeddings:
        model.tie_weights()
    repack_moe(model)
    remove_dispatch(model)
    return model, {
        "decoder": "compressed-tensors -> BF16; Transformers native Granite experts",
        "bits": scheme.weights.num_bits,
        "packed_modules": len(packed),
        "expert_projections": sum(".experts." in n for n in packed),
    }


def load_model(path, device):
    path = Path(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    source = path / "qda_source.json"
    if source.exists() and config.model_type == "granitemoe":
        model, meta = granite_packed(path, config)
        meta["source"] = json.loads(source.read_text())
    else:
        model, info = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            output_loading_info=True,
        )
        if any(
            info.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Incomplete checkpoint load: {info}")
        meta = {
            "decoder": "Transformers from_pretrained",
            "class": type(model).__name__,
        }
    model.eval().to(device)
    return model, meta
