"""Recode a single-shard BF16 model for dense TP1 CUTLASS W4AFP8."""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from safetensors import safe_open
from safetensors.torch import save_file


def recode(weight, scale, original, mode):
    # A row's groups share a channel factor. The serving kernel stores a
    # table of E4M3 products rather than sixteen equally spaced FP32 levels.
    values = weight.float().unflatten(-1, (-1, 128))
    channel = (scale.float().amax(-1, keepdim=True) / 448).clamp_min(1 / (448 * 512))
    norm = (scale.float() * channel.reciprocal()).clamp(max=448).to(torch.float8_e4m3fn)
    f8 = (norm.float() / 8).to(torch.float8_e4m3fn).float().unsqueeze(-1)
    channel = (channel * 8).unsqueeze(-1)
    levels = (torch.arange(-8, 8, device=weight.device) * f8).to(
        torch.float8_e4m3fn
    ).float() * channel
    step = f8 * channel
    # Equal-distance ties use round-to-even on the uniform grid. Zero groups
    # have identical served values for every nibble, so keep their original codes.
    uniform = (values / step.masked_fill(step == 0, 1)).round().clamp(-8, 7).long()
    closest = uniform
    if mode == "product":
        # Search the two bracketing levels directly. An FP32 midpoint can
        # round onto the input even when one endpoint is strictly closer.
        right = torch.searchsorted(levels.contiguous(), values.contiguous())
        for index in (right.clamp_max(15), (right - 1).clamp_min(0)):
            improve = (values - levels.gather(-1, index)).abs() < (
                values - levels.gather(-1, closest + 8)
            ).abs()
            closest = torch.where(improve, index - 8, closest)
    old = original.view_as(values).long()
    closest = torch.where(step == 0, old, closest)
    old_error = (values - levels.gather(-1, old + 8)).square()
    new_error = (values - levels.gather(-1, closest + 8)).square()
    if mode == "product" and (new_error > old_error).any():
        raise RuntimeError("Selected a worse served code")
    return closest.flatten(-2).to(torch.int8), torch.stack(
        (
            (closest != old).sum().double(),
            old_error.sum(dtype=torch.float64),
            new_error.sum(dtype=torch.float64),
            (step == 0).sum().double(),
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=["scale", "product"], required=True)
    args = p.parse_args()
    group = next(
        iter(
            json.loads((args.baseline / "config.json").read_text())[
                "quantization_config"
            ]["config_groups"].values()
        )
    )
    w, a = group["weights"], group["input_activations"]
    assert (
        w["num_bits"],
        w["group_size"],
        w["symmetric"],
        w["type"],
        a["type"],
        a["dynamic"],
    ) == (4, 128, True, "int", "float", True)
    args.out.mkdir(exist_ok=False)
    start = time.perf_counter()
    totals = torch.zeros(4, device="cuda", dtype=torch.float64)
    count = modules = 0
    with (
        safe_open(args.source / "model.safetensors", framework="pt") as src,
        safe_open(args.baseline / "model.safetensors", framework="pt") as base,
    ):
        state = {k: base.get_tensor(k) for k in base.keys()}
        metadata = base.metadata()
        for key in base.keys():
            if not key.endswith(".weight_packed"):
                continue
            prefix = key.removesuffix("weight_packed")
            weight = src.get_tensor(prefix + "weight").cuda()
            scale = state[prefix + "weight_scale"].cuda()
            assert weight.dtype == scale.dtype == torch.bfloat16
            original = unpack_from_int32(state[key].cuda(), 4, weight.shape)
            codes, stats = recode(weight, scale, original, args.mode)
            state[key] = pack_to_int32(codes, 4).cpu()
            totals += stats
            count += codes.numel()
            modules += 1
    save_file(state, args.out / "model.safetensors", metadata=metadata)
    for file in args.baseline.iterdir():
        if file.is_file() and file.suffix != ".safetensors":
            shutil.copyfile(file, args.out / file.name)
    changed, old_sse, new_sse, zero_groups = totals.tolist()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "modules": modules,
                "weights": count,
                "changed": int(changed),
                "zero_factor_groups": int(zero_groups),
                "old_mse": old_sse / count,
                "new_mse": new_sse / count,
                "recode_and_save_s": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )


if __name__ == "__main__":
    main()
