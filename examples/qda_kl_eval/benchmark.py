"""Matched resident-model timing and a same-position exact-KL diagnostic."""

import argparse
import importlib.metadata as md
import json
import platform
import time
from functools import partial
from pathlib import Path

import torch

from backend import checkpoint_hashes, load_model, tokenizer_info
from evaluate import (
    batches,
    binned,
    forward,
    identity,
    prepare,
    read_rows,
    score,
    sync,
    teacher_top,
    terms,
)
from make_candidates import REV, WEIGHTS

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"


def timed(fn, device):
    sync(device)
    start = time.perf_counter()
    value = fn()
    sync(device)
    return value, time.perf_counter() - start


@torch.inference_mode()
def online(teacher, candidate, rows, ident, size):
    total = torch.zeros((), device=teacher.device, dtype=torch.float64)
    for batch in batches(rows, size):
        ref = teacher_top(forward(teacher, batch, ident["tokenizer"]["pad_id"]), 128)
        total += binned(
            ref, forward(candidate, batch, ident["tokenizer"]["pad_id"])
        ).sum()
    return total.item() / ident["tokens_scored"]


@torch.inference_mode()
def exact_check(models, rows, pad, size):
    totals = {}
    for name, candidate in models.items():
        exact, approx = [], {k: [] for k in (128, 512, 2048)}
        tail = []
        for batch in batches(rows, size):
            p = forward(models["BF16"], batch, pad)
            q = forward(candidate, batch, pad)
            exact.append(terms(p, q).sum(-1))
            for k, values in approx.items():
                ref = teacher_top(p, k)
                values.append(binned(ref, q))
                if k == 128:
                    tail.append(ref["tail_logmass"].exp())
        ref_kl = torch.cat(exact)
        records = {}
        for k, values in approx.items():
            grouped = torch.cat(values)
            err = ref_kl - grouped
            if err.min().item() < -1e-11:
                raise ValueError("Data-processing inequality violated")
            records[k] = {
                "kl": grouped.mean().item(),
                "abs_mean_gap": err.mean().abs().item(),
                "max_per_position_gap": err.max().item(),
            }
        totals[name] = {
            "tokens": ref_kl.numel(),
            "exact": ref_kl.mean().item(),
            "by_k": records,
            "mean_teacher_tail_mass_k128": torch.cat(tail).mean().item(),
        }
    return totals


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("teacher", type=Path)
    p.add_argument("w4", type=Path)
    p.add_argument("w8", type=Path)
    p.add_argument("inputs", type=Path)
    p.add_argument(
        "artifacts", type=Path, help="A new directory on the cache filesystem"
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.repeats < 1:
        p.error("repeats must be positive")
    a.artifacts.mkdir()
    size, device = 2, a.device
    rows = read_rows(a.inputs)
    models, backend, loads = {}, {}, {}
    tok = tokenizer_info(a.teacher)
    for name, path in (("BF16", a.teacher), ("W4", a.w4), ("W8", a.w8)):
        if tokenizer_info(path) != tok:
            raise ValueError(f"Tokenizer mismatch: {name}")
        (models[name], backend[name]), loads[name] = timed(
            lambda: load_model(path, device), device
        )
        if name != "BF16" and backend[name]["bits"] != int(name[1:]):
            raise ValueError("Unexpected quantization scheme")
        forward(models[name], rows[:size], tok["pad_id"])
        sync(device)
    teacher = models["BF16"]
    ident = identity(rows, tok, MODEL_ID, REV, teacher.config.vocab_size)
    ident["reference_weights_sha256"] = checkpoint_hashes(a.teacher)
    if ident["reference_weights_sha256"] != WEIGHTS:
        raise ValueError("Unexpected teacher weights")
    candidates = ["W4", "W8"]
    trials = []
    for rep in range(a.repeats):
        cache = a.artifacts / f"cache-{rep}"
        record = {"rep": rep + 1, "cache_path": cache.name}
        # Reverse both strategy and candidate order across trials.
        order = candidates if rep % 2 == 0 else candidates[::-1]
        record["candidate_order"] = order
        modes = ("cached", "online") if rep % 2 == 0 else ("online", "cached")
        record["mode_order"] = modes
        for mode in modes:
            sync(device)
            start = time.perf_counter()
            build = 0.0
            if mode == "cached":
                _, build = timed(
                    lambda: prepare(teacher, rows, ident, cache, 128, size), device
                )
            stages, prefix, scores = {}, [], {}
            for name in order:
                if mode == "cached":
                    fn = partial(score, models[name], rows, ident, cache)
                else:
                    fn = partial(online, teacher, models[name], rows, ident, size)
                scores[name], stages[name] = timed(fn, device)
                prefix.append(time.perf_counter() - start)
            record[mode] = {
                "build_s": build,
                "candidate_s": stages,
                "prefix_wall_s": prefix,
                "kl": scores,
            }
        for name in candidates:
            if abs(record["online"]["kl"][name] - record["cached"]["kl"][name]) > 1e-11:
                raise ValueError("Online and cached scores differ")
        record["cache_bytes"] = sum(x.stat().st_size for x in cache.iterdir())
        trials.append(record)
        print(
            f"rep {rep + 1}",
            record["online"]["prefix_wall_s"],
            record["cached"]["prefix_wall_s"],
            flush=True,
        )
    subset = rows[:4]
    self_score = score(teacher, rows, ident, cache)
    exact = exact_check(models, subset, tok["pad_id"], size)
    from compressed_tensors.offload import remove_dispatch
    from llmcompressor.modeling.moe.linearize import linearize_moe, repack_moe

    before = forward(teacher, subset[:size], tok["pad_id"])
    repack_moe(linearize_moe(teacher))
    remove_dispatch(teacher)
    after = forward(teacher, subset[:size], tok["pad_id"])
    diff = (before - after).abs().max().item()
    if self_score != 0.0 or diff != 0.0:
        raise ValueError("Unexpected BF16 identity/round-trip error")
    result = {
        "base_commit": "f7d824cee12ff15bfa719ba8a7ea8ace7566df67",
        "identity": ident,
        "load_s": loads,
        "backend": backend,
        "batch_size": size,
        "trials": trials,
        "full_corpus_self_kl": self_score,
        "exact_first_four_sequences": exact,
        "bf16_linearize_repack_max_abs_logp_first_batch": diff,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "packages": {
                key: md.version(key)
                for key in (
                    "transformers",
                    "safetensors",
                    "compressed-tensors",
                    "numpy",
                    "pyarrow",
                    "accelerate",
                )
            },
        },
        "peak_allocated_bytes_three_resident_models": torch.cuda.max_memory_allocated(),
    }
    (a.artifacts / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
