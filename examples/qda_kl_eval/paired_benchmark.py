"""Matched, alternating-order measurements for the two paired.py cache formats."""

import argparse
import json
import time
from pathlib import Path

import torch

from backend import checkpoint_hashes, load_model, tokenizer_info
from benchmark import timed
from evaluate import batches, collate, read_rows, sync
from paired import Breakdown, Projection, full_kl, hidden, make_identity, prepare, score

HERE = Path(__file__).resolve().parent
PR_HEAD = "1ac89a7526b221b5780d91ed2e2da83a3fdf5c03"


def cache_size(cache, count):
    files = [p.stat() for p in cache.iterdir()]
    logical = sum(s.st_size for s in files)
    allocated = sum(s.st_blocks for s in files) * 512
    return {
        "file_bytes": logical,
        "allocated_bytes": allocated,
        "file_bytes_per_token": logical / count,
        "allocated_bytes_per_token": allocated / count,
    }


def native_check(models, heads, rows, pad, chunk):
    """Check split logits against unmodified forwards, including padding and scaling."""
    max_diff = dict.fromkeys(models, 0.0)
    different = dict.fromkeys(models, 0)
    native_kl = dict.fromkeys(models, 0.0)
    n = 0
    for group in batches(rows, 2):
        ids, mask, ri, ti = collate(group, pad, models["BF16"].device)
        for name, model in models.items():
            direct = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[
                ri, ti
            ]
            split = torch.cat(
                [heads[name].logits(s) for s in hidden(model, group, pad).split(chunk)]
            )
            max_diff[name] = max(max_diff[name], (direct - split).abs().max().item())
            different[name] += (direct != split).sum().item()
            logp = direct.double().log_softmax(-1)
            if name == "BF16":
                teacher = logp
            native_kl[name] += full_kl(teacher, logp).sum().item()
        n += len(ri)
    return {
        "native_fp64_kl": {k: v / n for k, v in native_kl.items()},
        "positions": n,
        "max_abs_logit_diff": max_diff,
        "unequal_logit_count": different,
    }


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("teacher", "w4", "w8", "inputs", "out"):
        p.add_argument(arg, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=8)
    a = p.parse_args()
    if a.repeats < 1:
        p.error("repeats must be positive")
    a.out.mkdir()
    prior = json.loads((HERE / "results.json").read_text())
    rows = read_rows(a.inputs)
    models, heads, loads, backend, fingerprints = {}, {}, {}, {}, {}
    pad, chunk = prior["identity"]["tokenizer"]["pad_id"], 128
    for name, path in (("BF16", a.teacher), ("W4", a.w4), ("W8", a.w8)):
        hashes = checkpoint_hashes(path)
        expected = (
            prior["identity"]["reference_weights_sha256"]
            if name == "BF16"
            else {
                "model.safetensors": prior["candidate_files"][name.lower()][
                    "model.safetensors_sha256"
                ]
            }
        )
        if hashes != expected or tokenizer_info(path) != prior["identity"]["tokenizer"]:
            raise ValueError(f"Unexpected {name} checkpoint or tokenizer")
        fingerprints[name] = hashes
        (models[name], backend[name]), loads[name] = timed(
            lambda: load_model(path, a.device), a.device
        )
        heads[name] = Projection(models[name])
        heads[name](hidden(models[name], rows[:2], pad))
    teacher = models["BF16"]
    ident = make_identity(
        teacher,
        rows,
        tokenizer_info(a.teacher),
        prior["identity"]["reference_model"],
        prior["identity"]["reference_revision"],
    )
    ident["reference_weights_sha256"] = fingerprints["BF16"]
    for field in ("inputs_sha256", "positions_sha256", "tokens_scored"):
        if ident[field] != prior["identity"][field]:
            raise ValueError(f"Input mismatch: {field}")
    # Read cache identity from disk on each score; every mode has the same warmed models.
    # Equal head hashes permit an independent candidate process to use its own head for P.
    trials = []
    for i in range(a.repeats):
        modes = ["hidden", "topk"] if i % 2 == 0 else ["topk", "hidden"]
        candidates = ["W4", "W8"] if (i // 2) % 2 == 0 else ["W8", "W4"]
        trial = {"repeat": i + 1, "mode_order": modes, "candidate_order": candidates}
        for kind in modes:
            cache = a.out / f"r{i+1}-{kind}"
            sync(a.device)
            start = time.perf_counter()
            _, build = timed(
                lambda: prepare(
                    teacher, heads["BF16"], rows, ident, cache, kind, chunk=chunk
                ),
                a.device,
            )
            stages, prefix, scores = {}, [], {}
            for name in candidates:
                scores[name], stages[name] = timed(
                    lambda: score(models[name], heads[name], rows, ident, cache, kind),
                    a.device,
                )
                prefix.append(time.perf_counter() - start)
            trial[kind] = {
                "build_s": build,
                "score_s": stages,
                "prefix_wall_s": prefix,
                "kl": scores,
                **cache_size(cache, ident["tokens_scored"]),
            }
        trials.append(trial)
        print(
            "repeat", i + 1, [(k, trial[k]["prefix_wall_s"]) for k in modes], flush=True
        )
    # Diagnostics run outside the measured trials. Stages may overlap across timelines.
    diagonals, diagnostic = {}, {}
    for kind in ("hidden", "topk"):
        cache = a.out / f"r{a.repeats}-{kind}"
        diagonals[kind] = score(teacher, heads["BF16"], rows, ident, cache, kind)
        diagnostic[kind] = {}
        for name in ("W4", "W8"):
            d = Breakdown()
            value, wall = timed(
                lambda: score(
                    models[name], heads[name], rows, ident, cache, kind, diag=d
                ),
                a.device,
            )
            diagnostic[kind][name] = {"kl": value, "wall_s": wall, **d.result()}
    oracle = native_check(models, heads, rows, pad, chunk)
    for name in ("W4", "W8"):
        exact = trials[-1]["hidden"]["kl"][name]
        if abs(oracle["native_fp64_kl"][name] - exact) > 1e-6:
            raise ValueError(f"Exact native parity failed: {name}, {oracle}, {exact}")
        if any(t["topk"]["kl"][name] > t["hidden"]["kl"][name] + 1e-6 for t in trials):
            raise ValueError(f"Coarse KL exceeds full KL: {name}")
    if max(abs(v) for v in diagonals.values()) > 1e-6:
        raise ValueError(f"Self KL failed: {diagonals}")
    result = {
        "pr_2663_head": PR_HEAD,
        "identity": ident,
        "model_load_s": loads,
        "projection_signatures": {k: v.signature for k, v in heads.items()},
        "checkpoint_sha256": fingerprints,
        "backend": backend,
        "native_check": oracle,
        "self_kl": diagonals,
        "trials": trials,
        "separate_stage_diagnostics": diagnostic,
        "batch_size": 2,
        "projection_chunk": chunk,
        "k": 128,
        "device": torch.cuda.get_device_name(a.device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "peak_allocated_bytes_including_diagnostics": torch.cuda.max_memory_allocated(
            a.device
        ),
    }
    (a.out / "paired-results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
