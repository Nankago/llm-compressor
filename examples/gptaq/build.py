"""Calibrate dense Qwen3 blocks sequentially and export a packed W4A16 checkpoint."""

import argparse
import contextlib
import hashlib
import json
import resource
import time
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized import (
    PackedQuantizationCompressor as Packed,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from inputs import TRAIN, WEIGHTS, digest, windows
from refit import refit
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from llmcompressor.modifiers.experimental.quantease import coordinate_sweep
from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight
from llmcompressor.observers.min_max import MemorylessMinMaxObserver

READERS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("self_attn.o_proj",),
    ("mlp.gate_proj", "mlp.up_proj"),
    ("mlp.down_proj",),
)


def run_layer(block, hidden, model, batch, capture):
    features, outputs = {}, []
    with contextlib.ExitStack() as stack:
        if capture:
            for group in READERS:
                name = group[0]
                features[name] = []

                def keep(_, args, key=name):
                    features[key].append(args[0].detach())

                stack.callback(
                    block.get_submodule(name).register_forward_pre_hook(keep).remove
                )
        for x in hidden.split(batch):
            pos = torch.arange(x.size(1), device=x.device)[None, :]
            mask = create_causal_mask(
                config=model.config,
                inputs_embeds=x,
                attention_mask=None,
                past_key_values=None,
                position_ids=pos,
            )
            rope = model.model.rotary_emb(x, pos)
            outputs.append(
                block(
                    x,
                    attention_mask=mask,
                    position_ids=pos,
                    position_embeddings=rope,
                    use_cache=False,
                )
            )
    return torch.cat(outputs), features


def stats(student, teacher):
    h = torch.zeros(student[0].size(-1), student[0].size(-1), device=student[0].device)
    cross = None if teacher is None else torch.zeros_like(h)
    count = 0
    for i, item in enumerate(student):
        x = item.flatten(0, -2).float()
        h.addmm_(x.T, x)
        if cross is not None:
            t = teacher[i].flatten(0, -2).float()
            if x.shape != t.shape:
                raise ValueError("Mismatched calibration positions")
            cross.addmm_((t - x).T, x)
        count += x.size(0)
    h.div_(count)
    if cross is not None:
        cross.div_(count)
    return h, cross


@torch.inference_mode()
def build(args):
    if sum(bool(v) for v in (args.sweep, args.alpha, args.float_prefix)) > 1:
        raise ValueError("Compare one mechanism at a time")
    if digest(args.model / "model.safetensors") != WEIGHTS:
        raise ValueError("Unexpected source weights")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids, meta = windows(args.train, TRAIN, tok, args.samples, args.length, args.seed)
    model, info = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Incomplete pretrained model: {info}")
    model.eval().to(args.device)
    if model.config.model_type != "qwen3" or model.model.has_sliding_layers:
        raise ValueError("Expected full-attention Qwen3")
    groups = {n for group in READERS for n in group}
    for block in model.model.layers:
        if groups != {
            n for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)
        }:
            raise ValueError("Unexpected decoder projection coverage")
    if args.out.exists():
        raise FileExistsError(args.out)
    qargs = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=128,
        actorder="static",
        observer="memoryless_minmax",
    )
    scheme = QuantizationScheme(
        targets=["Linear"], weights=qargs, format="pack-quantized"
    )

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    packed, layer_times = {}, []
    student = model.model.embed_tokens(ids.to(args.device))
    teacher = student if args.alpha else None
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync()
    start = time.perf_counter()
    for index, block in enumerate(model.model.layers):
        layer_start = time.perf_counter()
        if teacher is not None:
            teacher, tf = run_layer(block, teacher, model, args.batch, True)
        else:
            tf = None
        original_output, sf = run_layer(block, student, model, args.batch, True)
        for group in READERS:
            h, cross = stats(
                sf.pop(group[0]), tf.pop(group[0]) if tf is not None else None
            )
            if index == 0 and cross is not None and torch.count_nonzero(cross):
                raise ValueError(
                    "The two original streams diverged before quantization"
                )
            for name in group:
                mod = block.get_submodule(name)
                fitted = refit(mod.weight, h, cross, args.alpha)
                qp = MemorylessMinMaxObserver("weight", qargs)(fitted).get_qparams()
                scale = qp["scale"]
                q, _, fallback = quantize_weight(
                    fitted.float().unsqueeze(0),
                    h.clone().unsqueeze(0),
                    scale.unsqueeze(0),
                    qp["zero_point"].unsqueeze(0),
                    None,
                    qargs,
                    128,
                    0.01,
                )
                if fallback.any():
                    raise ArithmeticError("Unexpected GPTQ RTN fallback")
                value = q[0].to(mod.weight.dtype)
                if args.sweep:
                    value = coordinate_sweep(mod.weight, value, h, scale, 128)
                state = Packed.compress(
                    {"weight": value, "weight_scale": scale}, scheme
                )
                if not torch.equal(Packed.decompress(state, scheme)["weight"], value):
                    raise ValueError(
                        "Packed codes/scales do not reproduce the fitted weight"
                    )
                mod.weight.copy_(value)
                prefix = f"model.layers.{index}.{name}"
                for suffix, tensor in state.items():
                    packed[f"{prefix}.{suffix}"] = tensor.cpu().contiguous()
        if args.float_prefix:
            student = original_output
        else:
            student, _ = run_layer(block, student, model, args.batch, False)
        sync()
        layer_times.append(time.perf_counter() - layer_start)
        print(
            f"layer {index + 1}/{len(model.model.layers)}: {layer_times[-1]:.2f} s",
            flush=True,
        )
    seconds = time.perf_counter() - start
    export_start = time.perf_counter()
    ignore = {
        k.removesuffix(".weight_packed") + ".weight"
        for k in packed
        if k.endswith(".weight_packed")
    }
    if len(ignore) != 7 * len(model.model.layers):
        raise AssertionError("Missing quantized projections")
    first_hash = hashlib.sha256()
    for key, value in sorted(packed.items()):
        if key.startswith("model.layers.0."):
            first_hash.update(key.encode())
            first_hash.update(value.view(torch.uint8).numpy().tobytes())
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise AssertionError("Expected tied embeddings")
    for key, value in model.state_dict().items():
        if key not in ignore and key != "lm_head.weight":
            packed[key] = value.cpu().contiguous()
    args.out.mkdir(parents=True)
    save_file(packed, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    model.config.quantization_config = QuantizationConfig(
        config_groups={"group_0": scheme},
        format="pack-quantized",
        quantization_status="compressed",
        ignore=["lm_head"],
    ).model_dump(mode="json")
    model.config.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    result = dict(
        alpha=args.alpha,
        sweep=args.sweep,
        float_prefix=args.float_prefix,
        train=meta,
        device=args.device,
        threads=torch.get_num_threads(),
        quantization_s=seconds,
        layer_seconds=layer_times,
        export_s=time.perf_counter() - export_start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        packed_projections=len(ignore),
        first_block_sha256=first_hash.hexdigest(),
        checkpoint_sha256=digest(args.out / "model.safetensors"),
        checkpoint_bytes=(args.out / "model.safetensors").stat().st_size,
    )
    (args.out / "build.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "train", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--float-prefix",
        action="store_true",
        help="Propagate pre-quantization block outputs",
    )
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--alpha", required=True, type=float)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    build(p.parse_args())
