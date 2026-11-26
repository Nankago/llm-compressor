"""Verify the OneShot recipe and packed export on pretrained Qwen blocks."""

import argparse
import json
import time
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from inputs import TRAIN, WEIGHTS, digest, windows
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.experimental.quantease import QuantEaseModifier
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.recipe import Recipe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "train", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--full",
        action="store_true",
        help="Target every decoder projection at 32 x 128 tokens",
    )
    a = p.parse_args()
    assert digest(a.model / "model.safetensors") == WEIGHTS
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    count, length = (32, 128) if a.full else (8, 64)
    tokens, meta = windows(a.train, TRAIN, tok, count, length, 42)
    target = r"re:model\.layers\.0\.(self_attn|mlp)\..*_proj$"
    if a.full:
        target = r"re:model\.layers\.[0-9]+\.(self_attn|mlp)\..*_proj$"
    scheme = QuantizationScheme(
        targets=[target],
        weights=QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="group",
            group_size=128,
            type="int",
            observer="memoryless_minmax",
        ),
    )
    for name, cls in (("stock", GPTQModifier), ("sweep", QuantEaseModifier)):
        directory = a.out / name
        if directory.exists():
            raise FileExistsError(directory)
        model = AutoModelForCausalLM.from_pretrained(
            a.model,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        loader = DataLoader(
            [{"input_ids": x, "attention_mask": torch.ones_like(x)} for x in tokens],
            batch_size=1,
        )
        recipe = Recipe.from_dict(
            {
                "fit_stage": {
                    "quant_modifiers": {
                        cls.__name__: {
                            "config_groups": {
                                "group_0": scheme.model_dump(mode="json")
                            },
                            "batched_quantization": None,
                        }
                    }
                }
            }
        )
        assert isinstance(recipe.modifiers[0], cls)
        start = time.perf_counter()
        oneshot(
            model=model,
            tokenizer=tok,
            recipe=recipe,
            dataset=loader,
            pipeline="sequential",
            sequential_targets=["Qwen3DecoderLayer"],
            num_calibration_samples=len(tokens),
            max_seq_length=length,
            batch_size=1,
            shuffle_calibration_samples=False,
            propagate_error=True,
            output_dir=None,
        )
        seconds = time.perf_counter() - start
        model.save_pretrained(directory, save_compressed=True)
        if a.full:
            record = dict(
                alpha=0,
                sweep=name == "sweep",
                train=meta,
                quantization_s=seconds,
                packed_projections=196,
                checkpoint_sha256=digest(directory / "model.safetensors"),
            )
            (directory / "build.json").write_text(json.dumps(record, indent=2) + "\n")
        del model
        checked, info = AutoModelForCausalLM.from_pretrained(
            directory,
            dtype=torch.bfloat16,
            quantization_config=CompressedTensorsConfig(dequantize=True),
            local_files_only=True,
            attn_implementation="eager",
            output_loading_info=True,
        )
        assert not any(
            info.get(k)
            for k in (
                "unexpected_keys",
                "missing_keys",
                "mismatched_keys",
                "error_msgs",
            )
        )
        with torch.inference_mode():
            assert torch.isfinite(checked(tokens[:1], use_cache=False).logits).all()
        del checked
    with (
        safe_open(a.out / "stock/model.safetensors", framework="pt") as s,
        safe_open(a.out / "sweep/model.safetensors", framework="pt") as t,
    ):
        assert set(s.keys()) == set(t.keys())
        packed = [n for n in s.keys() if n.endswith(".weight_packed")]
        assert len(packed) == (196 if a.full else 7)
        changed = sum(not torch.equal(s.get_tensor(n), t.get_tensor(n)) for n in packed)
        assert changed and all(
            torch.equal(s.get_tensor(n), t.get_tensor(n))
            for n in s.keys()
            if n not in packed
        )
    (a.out / "native.json").write_text(
        json.dumps(
            dict(
                inputs=meta,
                packed_modules=len(packed),
                modules_with_different_codes=changed,
                all_other_tensors_match=True,
                both_exports_load_and_forward=True,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
