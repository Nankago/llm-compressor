"""Build the compact fork artifact from verified raw idle_observer.py outputs."""

import argparse
import json
import os
from pathlib import Path

from summarize_observer_trace import analyze

HERE = Path(__file__).resolve().parent


def collect(run_dir):
    def read(name):
        return json.loads((run_dir / name).read_text())

    preflight = read("preflight.json")
    gpu = preflight["selected_gpu"]
    if not gpu:
        raise ValueError("No idle GPU selected")
    shared = json.loads((HERE / "results/runs.json").read_text())["observer-auto"]
    runs = []
    for label in ("warmup", "run1", "run2", "run3", "trace"):
        raw = read(f"{label}.json")
        for key in (
            "upstream",
            "revision",
            "weight_sha256",
            "input_ids_sha256_le_i64",
            "samples",
            "sequence_length",
            "batch_histogram",
            "calls",
            "mode",
        ):
            if raw[key] != shared[key]:
                raise ValueError(f"{label} mismatch: {key}")
        row = {
            "run": label,
            "oneshot_s": raw["oneshot_wall_s"],
            "weight_observe_s": raw["cuda_event_s"]["weight_observe"],
        }
        for when in ("before", "after"):
            snap = read(f"{label}-{when}.json")
            device = next(d for d in snap["devices"] if d["uuid"] == gpu)
            if device["processes"] or (when == "before" and device["util_pct"]):
                raise ValueError(f"Device contention at {label}/{when}")
            row[when] = {"utc": snap["utc"], **device}
        runs.append(row)

    trace = analyze(run_dir / "observer-trace.sqlite")
    summary = read("timings.json")["summary"]
    observer = trace["GPTQ/weight_observe"]
    validator = trace["GPTQ/_get_validated_importance"]
    return {
        "benchmark_checkout": preflight["base_commit"],
        "nsys": preflight["nsys"],
        "environment": raw["environment"],
        "model_revision": raw["revision"],
        "input_ids_sha256_le_i64": raw["input_ids_sha256_le_i64"],
        "runs": runs,
        "baseline_summary_s": summary,
        "shared_gpu_reference_s": {
            "observer_auto": shared["cuda_event_s"]["weight_observe"],
            "oneshot_auto": shared["oneshot_wall_s"],
        },
        "server_trace_relative_to_original_checkout": os.path.relpath(
            run_dir / "observer-trace.nsys-rep",
            os.environ.get("OBSERVER_ORIGINAL_CHECKOUT", HERE.parents[1]),
        ),
        "finding": (
            f"The three unprofiled idle-H100 runs had a median weight_observe of "
            f"{summary['weight_observe_s']['median']:.3f} s. Inside the separate "
            f"weight_observe NVTX windows, kernels were active for "
            f"{observer['kernel_active_ms']:.1f} ms and synchronous CUDA API calls "
            f"occupied {observer['sync_api_residence_ms']:.1f} ms of host time. "
            f"The nested validator's API residence was {validator['sync_api_residence_ms']:.1f} ms."
        ),
        "trace": trace,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("result_file", type=Path)
    args = parser.parse_args()
    result = collect(args.run_dir.resolve())
    args.result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(result["finding"])
