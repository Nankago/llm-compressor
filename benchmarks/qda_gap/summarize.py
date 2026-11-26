"""Reduce the run directory to portable paired metrics and verify packed artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import binomtest


def load(path):
    return json.loads(path.read_text())


def last_json(path):
    return json.loads(
        next(
            line
            for line in reversed(path.read_text().splitlines())
            if line.startswith("{")
        )
    )


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tensor_audit(root, left, right):
    file = "model.safetensors"
    counts = {"total": 0, "packed": 0, "changed_packed": 0, "changed_other": 0}
    with (
        safe_open(root / left / file, framework="pt") as a,
        safe_open(root / right / file, framework="pt") as b,
    ):
        assert a.keys() == b.keys()
        for k in a.keys():
            x, y = a.get_tensor(k), b.get_tensor(k)
            assert x.shape == y.shape and x.dtype == y.dtype
            changed = not torch.equal(x, y)
            packed = k.endswith(".weight_packed")
            counts["total"] += 1
            counts["packed"] += int(packed)
            counts["changed_packed" if packed else "changed_other"] += int(changed)
    assert counts["packed"] and counts["changed_other"] == 0
    if right == "gptq-control":
        assert counts["changed_packed"] == 0
    assert (root / left / "config.json").read_bytes() == (
        root / right / "config.json"
    ).read_bytes()
    return counts


def paired(root, left, right):
    a, b = [load(root / f"{k}-full.json") for k in (left, right)]
    assert a["tokens_sha256"] == b["tokens_sha256"]
    assert [(r["row"], r["positions"]) for r in a["rows"]] == [
        (r["row"], r["positions"]) for r in b["rows"]
    ]
    n = np.array([r["positions"] for r in a["rows"]])
    ix = np.random.default_rng(13).integers(len(n), size=(10000, len(n)))
    result = {}
    for key in ("kl", "nll"):
        d = np.array(
            [
                y[key + "_sum"] - x[key + "_sum"]
                for x, y in zip(a["rows"], b["rows"], strict=True)
            ]
        )
        ci = np.quantile(d[ix].sum(1) / n[ix].sum(1), [0.025, 0.975])
        result[key] = {
            "delta": float(d.sum() / n.sum()),
            "paired_paragraph_ci95": ci.tolist(),
        }
    if all((root / f"{x}-arc.json").exists() for x in (left, right)):
        c, d = [
            np.array(load(root / f"{x}-arc.json")["correct_by_row"])
            for x in (left, right)
        ]
        wins, losses = int((d & ~c).sum()), int((c & ~d).sum())
        result["arc"] = {
            "gained": wins,
            "lost": losses,
            "delta_correct": wins - losses,
            "mcnemar_exact_two_sided_p": binomtest(wins, wins + losses).pvalue
            if wins + losses
            else 1.0,
        }
    return result


def metrics(root, arm):
    f = load(root / f"{arm}-full.json")
    result = {
        "full_kl": f["kl"],
        "nll": f["nll"],
        "positions": sum(r["positions"] for r in f["rows"]),
        "paragraphs": len(f["rows"]),
        "max_native_logprob_error": f["max_logprob_difference"],
        "serving_kernels": f["bound"],
    }
    score, arc = root / f"{arm}-score.json", root / f"{arm}-arc.json"
    if score.is_file():
        result["single_anchor_exact_kl"] = load(score)["kl_mean"]
    if arc.is_file():
        m = load(arc)
        result["arc_correct"], result["arc_total"] = m["correct"], m["total"]
    file = root / arm / "model.safetensors"
    if file.is_file():
        result.update(checkpoint_bytes=file.stat().st_size, checkpoint_sha256=sha(file))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    root = a.root
    torch.set_num_threads(2)
    arms = [
        "ref",
        "packed",
        "final-scale",
        "final-product",
        "gptq-main",
        "gptq-aware",
        "four-ref",
        "four-main",
        "four-aware",
        "seed42-main",
        "seed42-aware",
    ]
    pairs = [
        ("packed", "final-scale"),
        ("final-scale", "final-product"),
        ("packed", "final-product"),
        ("gptq-main", "gptq-aware"),
        ("four-main", "four-aware"),
        ("seed42-main", "seed42-aware"),
    ]
    operator = last_json(root / "probe-serve.log")
    data = {
        "upstream": "18b10e2d7269e332c05eef4dda52adf74c08872e",
        "qda": "dc39a02bfd87bb2790e34001708d54ce9131e076",
        "vllm": "62d4bc3c7e33b1b6c906e233e6af2d3506c482ca",
        "compressed_tensors": "0.20.2a20261123",
        "compressed_tensors_source": "7be5968f0b64dfe73a189f0cf128f17c0a9c06ca",
        "w4_gpu_uuid": "GPU-" + operator["gpu_uuid"].removeprefix("GPU-"),
        "held_out_ids_sha256": sha(root / "tokens.json"),
        "operator": operator,
        "pretrained_rank_local": load(root / "row-shards.json"),
        "models": {
            "Qwen/Qwen3-0.6B": {
                "sha": "c1899de289a04d12100db370d81485cdf75e47ca",
                "files": {
                    "model.safetensors": (
                        "f47f71177f32bcd101b7573ec9171e6a57f4f4d3"
                        "1148d38e382306f42996874b"
                    )
                },
            },
            "Qwen/Qwen3-4B": load(root / "model4.json"),
            "Qwen/Qwen3.5-0.8B-Base": load(root / "model35.json"),
        },
        "scores": {a: metrics(root, a) for a in arms},
        "paired_right_minus_left": {f"{x}/{y}": paired(root, x, y) for x, y in pairs},
        "tensor_checks": {
            f"{x}/{y}": tensor_audit(root, x, y)
            for x, y in pairs + [("gptq-main", "gptq-control")]
        },
        "build_cost": {
            a: last_json(root / f"{a}.log")
            for a in [
                "packed",
                "final-product",
                "final-scale",
                "gptq-main",
                "gptq-control",
                "gptq-aware",
                "four-main",
                "four-aware",
                "seed42-main",
                "seed42-aware",
            ]
        },
        "mapping": last_json(root / "mapping-control.log"),
        "tiny_moe": [],
    }
    for r in ("r1", "r2"):
        x = load(root / f"tiny-{r}.json")
        dev = load(root / f"tiny-{r}.device.json")
        assert not dev["foreign_pids"] and dev["returncode"] == 0
        data["tiny_moe"].append(
            {
                k: x[k]
                for k in [
                    "revision",
                    "weight_sha256",
                    "gpu_before",
                    "samples",
                    "sequence_length",
                    "input_ids_sha256_le_i64",
                    "oneshot_wall_s",
                    "peak_allocated_bytes",
                    "cuda_event_s",
                    "batch_histogram",
                    "module_count_by_cholesky",
                    "rtn_fallback_modules",
                ]
            }
            | {
                "runtime_llmc": x["environment"]["packages"]["llmcompressor"],
                "foreign_pids": dev["foreign_pids"],
            }
        )
    for left, right in pairs[-3:]:
        assert (
            data["build_cost"][left]["calibration_sha256"]
            == data["build_cost"][right]["calibration_sha256"]
        )
    assert (
        data["build_cost"]["seed42-main"]["calibration_sha256"]
        != data["build_cost"]["gptq-main"]["calibration_sha256"]
    )
    a.out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
