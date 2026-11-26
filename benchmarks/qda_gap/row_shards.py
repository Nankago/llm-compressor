"""Compare TP1 and two-way rank-local RTN code selection on real Qwen3 rows."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from kernel_probe import apply
from qda_quant.solve.quant import round_fp8
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


def unpack(words, shift):
    return ((((words[..., None] >> shift) & 15) - 8).flatten(1)).float()


def pack(codes, shift):
    return (((codes.int() + 8).unflatten(1, (-1, 8)) << shift).sum(-1)).int()


def examine(weight, scale, stock_words, full_words):
    out, width = weight.shape
    local_width = width // 2
    assert local_width % 128 == 0
    shift = torch.arange(8, device="cuda") * 4
    positions = torch.linspace(0, local_width - 1, steps=16, device="cuda").long()
    basis = torch.zeros(
        (len(positions), local_width), dtype=torch.bfloat16, device="cuda"
    )
    basis.scatter_(1, positions[:, None], 1)
    qx, sx = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)(basis)
    active = qx.float().gather(1, positions[:, None])
    assert (active == 448).all() and (qx.float().count_nonzero(1) == 1).all()
    totals = dict(
        weights=0,
        strictly_improved_over_full=0,
        full_worse_than_stock=0,
        changed_codes=0,
        basis_values=0,
        basis_differences=0,
        scales_first_differences=0,
        token_first_differences=0,
        channel_first_differences=0,
        actual_local_better_than_full=0,
        actual_local_worse_than_full=0,
        actual_pairs=0,
        stock_sse=0.0,
        full_sse=0.0,
        local_sse=0.0,
    )
    for rank in range(2):

        def shard(x):
            return x.chunk(2, dim=1)[rank].contiguous()

        s = shard(scale)
        original = shard(weight).float()
        stock, full = shard(stock_words), shard(full_words)
        actual_stock, (f8, channel), kernel = apply(stock, s, basis, global_k=width)
        assert kernel == "CutlassW4A8LinearKernel"
        factor = f8.float().repeat_interleave(128, dim=1)
        local, local_level = round_fp8(original, factor, channel, -8, 7)
        errors = []
        actual_errors = []
        for words, key, actual in [
            (stock, "stock", actual_stock),
            (full, "full", None),
            (pack(local, shift), "local", None),
        ]:
            if actual is None:
                actual, _, _ = apply(words, s, basis, global_k=width)
            codes = unpack(words, shift)
            levels = (codes * factor).to(torch.float8_e4m3fn).float() * channel
            if key == "local":
                assert torch.equal(levels, local_level)
            actual_errors.append(
                (actual.float() - original.index_select(1, positions).T).abs()
            )
            observed = levels.index_select(1, positions).T.to(torch.bfloat16)
            totals["basis_values"] += actual.numel()
            totals["basis_differences"] += int((observed != actual).sum())
            products = (
                (codes.index_select(1, positions) * factor.index_select(1, positions))
                .to(torch.float8_e4m3fn)
                .float()
                .T
            )
            acc = products * active
            for name, predicted in (
                ("scales_first", acc * (sx * channel.T)),
                ("token_first", (acc * sx) * channel.T),
                ("channel_first", (acc * channel.T) * sx),
            ):
                totals[name + "_differences"] += int(
                    (actual != predicted.to(torch.bfloat16)).sum()
                )
            err = (levels - original).abs()
            totals[key + "_sse"] += err.square().sum(dtype=torch.float64).item()
            errors.append(err)
        totals["actual_local_better_than_full"] += int(
            (actual_errors[2] < actual_errors[1]).sum()
        )
        totals["actual_local_worse_than_full"] += int(
            (actual_errors[2] > actual_errors[1]).sum()
        )
        totals["actual_pairs"] += actual_errors[2].numel()
        assert torch.all(errors[2] <= errors[1]) and torch.all(errors[2] <= errors[0])
        totals["strictly_improved_over_full"] += int((errors[2] < errors[1]).sum())
        totals["full_worse_than_stock"] += int((errors[1] > errors[0]).sum())
        totals["changed_codes"] += int((unpack(full, shift) != local).sum())
        totals["weights"] += out * local_width
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "stock", "full", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    result = {}
    modules = 0
    with (
        safe_open(args.source / "model.safetensors", framework="pt") as ref,
        safe_open(args.stock / "model.safetensors", framework="pt") as stock,
        safe_open(args.full / "model.safetensors", framework="pt") as full,
    ):
        for name in stock.keys():
            if not name.endswith((".o_proj.weight_packed", ".down_proj.weight_packed")):
                continue
            prefix = name.removesuffix("weight_packed")
            scale = prefix + "weight_scale"
            assert torch.equal(stock.get_tensor(scale), full.get_tensor(scale))
            row = examine(
                ref.get_tensor(prefix + "weight").cuda(),
                stock.get_tensor(scale).cuda(),
                stock.get_tensor(name).cuda(),
                full.get_tensor(name).cuda(),
            )
            group = result.setdefault(prefix.split(".")[-2], {})
            for key, count in row.items():
                group[key] = group.get(key, 0) + count
            modules += 1
    assert modules == 56
    for row in result.values():
        assert row["token_first_differences"] == 0
        for key in ("stock", "full", "local"):
            row[key + "_mse"] = row.pop(key + "_sse") / row["weights"]
    args.out.write_text(
        json.dumps(
            dict(
                modules=modules,
                shard_ranks_per_module=2,
                gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                projections=result,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory, set_current_vllm_config(
        VllmConfig()
    ):
        init_distributed_environment(1, 0, f"file://{directory}/rendezvous", 0)
        initialize_model_parallel()
        try:
            with torch.inference_mode():
                main()
        finally:
            cleanup_dist_env_and_memory()
