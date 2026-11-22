"""Build two standard compressed-tensors checkpoints from identical training tokens."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.gptq import GPTQModifier
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend import sha_file

REV = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
WEIGHTS = {
    "model-00001-of-00002.safetensors": "6933b6375074dc6cc4c0b98c9e126cd5fc899ef0cd80db0a932a232c6393c9cd",
    "model-00002-of-00002.safetensors": "271870f88e36f28ca87e86e11fa971b0b3e268fb9ac4495d42c66fdd1bc9d060",
}
CAL = "7bb325175365a1c8280294795a5819e44cb9a2755df536704f9ad21c94da4ac0"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("calibration_bank", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("scheme", choices=["W4A16", "W8A16"])
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if any(sha_file(a.source / name) != sha for name, sha in WEIGHTS.items()):
        raise ValueError("Unexpected Granite BF16 checkpoint")
    tokens = np.load(a.calibration_bank, allow_pickle=False)[:8, :128].copy()
    if hashlib.sha256(tokens.astype("<i8").tobytes()).hexdigest() != CAL:
        raise ValueError("Wrong Granite calibration tokens")
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(a.source, local_files_only=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        a.source,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    if any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(info)
    linearize_moe(model.eval())
    rows = [
        {"input_ids": t, "attention_mask": torch.ones_like(t)}
        for t in torch.from_numpy(tokens)
    ]
    modifier = GPTQModifier(
        scheme=a.scheme,
        targets="Linear",
        ignore=["lm_head", r"re:.*\.gate$"],
        batched_quantization="auto",
        block_size=128,
        dampening_frac=0.01,
    )
    oneshot(
        model=model,
        tokenizer=tok,
        dataset=DataLoader(rows, batch_size=1),
        recipe=modifier,
        pipeline="sequential",
        sequential_targets_per_subgraph=1,
        batch_size=1,
        max_seq_length=128,
        num_calibration_samples=8,
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=True,
        propagate_error=True,
        tie_word_embeddings=False,
        output_dir=None,
    )
    model.save_pretrained(a.output, save_compressed=True, save_original_format=False)
    tok.save_pretrained(a.output)
    (a.output / "qda_source.json").write_text(
        json.dumps(
            {
                "model": "ibm-granite/granite-3.1-1b-a400m-base",
                "revision": REV,
                "calibration_sha256": CAL,
                "scheme": a.scheme,
            },
            indent=2,
        )
        + "\n"
    )
