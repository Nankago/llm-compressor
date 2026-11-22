"""Reduce the raw run.py outputs without copying per-module dumps or private inputs."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def device(path):
    d = read(path)
    if (
        d["returncode"]
        or d["foreign_pids"]
        or d["before"]["pids"]
        or d["after"]["pids"]
    ):
        raise ValueError(f"Failed or contended run: {path}")
    return {k: v for k, v in d.items() if k != "pid"}


def timing(path):
    r = read(path)
    counts = r["calls"]
    s = r["cuda_event_s"]
    return {
        "start_utc": r["start_utc"],
        "propagate_error": r["propagate_error"],
        "log_sequential_error": r["log_sequential_error"],
        "oneshot_s": r["oneshot_wall_s"],
        "first_s": s["calib_forward_including_hooks"],
        "second_s": s.get("propagate_forward", 0.0),
        "observer_s": s["weight_observe"],
        "peak_allocated_bytes": r["peak_allocated_bytes"],
        "peak_reserved_bytes": r["peak_reserved_bytes"],
        "first_calls": counts["calib_forward_including_hooks"],
        "second_calls": counts.get("propagate_forward", 0),
        "observer_calls": counts["weight_observe"],
        "fallbacks": len(r["rtn_fallback_modules"]),
        "setup_s": r["setup_wall_s"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    w = a.raw
    d = {
        "source_commit": "c81852cbfc04fca46f5f85a9d88a60525efb4385",
        "upstream_base": "f7d824cee12ff15bfa719ba8a7ea8ace7566df67",
        "kl_source_commit": "8e7e6ad781a416b967b3c4a498d8469d97224d61",
    }
    for model, n, label in [("granite", 4, "granite"), ("tinymoe", 2, "tiny")]:
        root = w / model
        ref = read(root / "r1-on.json")
        modeldata = {
            k: ref[k]
            for k in (
                "model",
                "revision",
                "weight_sha256",
                "seed",
                "samples",
                "sequence_length",
                "input_ids_sha256_le_i64",
                "module_count_by_cholesky",
                "observer_source_sha256",
                "gpu_before",
            )
        }
        rows = []
        for i in range(n + 1):
            arm_order = ["on", "off"] if i % 2 else ["off", "on"]
            pair = {"pair": i, "warmup": i == 0, "order": arm_order}
            for arm in arm_order:
                path = root / f"r{i}-{arm}.json"
                rr = read(path)
                if (
                    rr["input_ids_sha256_le_i64"] != ref["input_ids_sha256_le_i64"]
                    or rr["observer_source_sha256"] != ref["observer_source_sha256"]
                ):
                    raise ValueError("Input or observer changed during comparison")
                pair[arm] = {
                    **timing(path),
                    "device": device(root / f"r{i}-{arm}.device.json"),
                }
            pair["wall_saved_s"] = pair["on"]["oneshot_s"] - pair["off"]["oneshot_s"]
            pair["wall_saved_percent"] = (
                100 * pair["wall_saved_s"] / pair["on"]["oneshot_s"]
            )
            rows.append(pair)
        modeldata["runs"] = rows
        modeldata["quality"] = read(w / f"{label}-quality.json")
        d[model] = modeldata
    d["environment"] = ref["environment"]
    d["sqnr_enabled_control"] = timing(w / "sqnr-check.json")
    d["sampled_forwards"] = read(w / "trace/observations.json")
    d["trace"] = read(w / "trace-summary.json")
    d["gram_replay"] = read(w / "gram-replay.json")
    d["diagnostic_devices"] = {
        name: device(w / f"{name}.device.json")
        for name in (
            "quality-granite",
            "quality-tiny-v2",
            "sqnr-check",
            "trace-granite",
            "gram-replay",
        )
    }
    a.output.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
