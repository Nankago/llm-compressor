"""Score exported W4A16 checkpoints against the BF16 teacher on held-out articles."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from compressed_tensors.compressors.pack_quantized import (
    PackedQuantizationCompressor as Packed,
)
from compressed_tensors.quantization import QuantizationScheme
from inputs import TEST, WEIGHTS, digest, windows
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig


def load(path, device, quantized):
    opts = (
        {"quantization_config": CompressedTensorsConfig(dequantize=True)}
        if quantized
        else {}
    )
    model, info = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
        local_files_only=True,
        **opts,
    )
    if any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"Incomplete model: {info}")
    if quantized:
        cfg = json.loads((path / "config.json").read_text())["quantization_config"]
        scheme = QuantizationScheme.model_validate(cfg["config_groups"]["group_0"])
        with safe_open(path / "model.safetensors", framework="pt") as f:
            names = [
                key.removesuffix(".weight_packed")
                for key in f.keys()
                if key.endswith(".weight_packed")
            ]
            assert len(names) == 196
            for name in names:
                state = {
                    k: f.get_tensor(name + "." + k)
                    for k in ("weight_packed", "weight_shape", "weight_scale")
                }
                value = Packed.decompress(state, scheme)["weight"]
                if not torch.equal(model.get_submodule(name).weight.cpu(), value):
                    raise ValueError(f"Loaded weights differ: {name}")
    return model.eval().to(device)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("model", "test", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--candidates", nargs="+", type=Path, required=True)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if digest(a.model / "model.safetensors") != WEIGHTS:
        raise ValueError("Teacher checkpoint mismatch")
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    ids, meta = windows(a.test, TEST, tok, a.samples, a.length, 7651)
    models = {"bf16": load(a.model, a.device, False)}
    builds = {}
    for path in a.candidates:
        record = json.loads((path / "build.json").read_text())
        if digest(path / "model.safetensors") != record["checkpoint_sha256"]:
            raise ValueError("Candidate checkpoint mismatch")
        builds[path.name] = record
        models[path.name] = load(path, a.device, True)
    rows = {k: [] for k in models}
    for seq in ids:
        x = seq[None, :].to(a.device)
        for key, model in models.items():
            q = model(x, use_cache=False).logits[0, :-1].float().log_softmax(-1)
            if key == "bf16":
                teacher = q
                prob = q.exp()
            kl = (prob * (teacher - q)).sum(-1, dtype=torch.float64).mean().item()
            nll = -q.gather(-1, x[0, 1:, None]).double().mean().item()
            rows[key].append([kl, nll])
    draws = np.random.default_rng(20261126).integers(len(ids), size=(10000, len(ids)))
    quality, pairs = {}, {}
    for name, v in rows.items():
        raw = np.array(v)
        kl, nll = raw.mean(0)
        quality[name] = dict(kl=kl, nll=nll, ppl=math.exp(nll), article_kl_nll=v)
        if name == "bf16" or not (
            builds[name]["alpha"]
            or builds[name].get("sweep")
            or builds[name].get("float_prefix")
        ):
            continue
        controls = [
            n
            for n, b in builds.items()
            if not b["alpha"]
            and not b.get("sweep")
            and not b.get("float_prefix")
            and b["train"] == builds[name]["train"]
        ]
        if len(controls) != 1:
            raise ValueError("Expected one matched stock checkpoint")
        control = controls[0]
        if (
            builds[name]["alpha"]
            and builds[name]["first_block_sha256"]
            != builds[control]["first_block_sha256"]
        ):
            raise ValueError(
                "GPTAQ with identical first-block inputs diverged from stock"
            )
        delta = raw - np.asarray(rows[control])
        pairs[name] = dict(
            control=control,
            delta_kl=delta[:, 0].mean(),
            delta_nll=delta[:, 1].mean(),
            ci95_kl_nll=np.quantile(
                delta[draws].mean(1), [0.025, 0.975], axis=0
            ).tolist(),
            articles_lower_kl=int((delta[:, 0] < 0).sum()),
        )
    assert quality["bf16"]["kl"] == 0
    a.out.write_text(
        json.dumps(
            dict(
                test=meta,
                positions=len(ids) * (a.length - 1),
                builds=builds,
                quality=quality,
                pairs=pairs,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
