"""Attribute Kineto device activity to the CPU range that submitted it."""

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path


def union_us(intervals):
    end = float("-inf")
    total = 0.0
    for lo, hi in sorted(intervals):
        total += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return total


def summarize(path):
    events = [
        e
        for e in json.loads(path.read_text())["traceEvents"]
        if e.get("ph") == "X" and "dur" in e
    ]
    tags = defaultdict(list)
    for e in events:
        if e["name"].startswith("ablation/") and e.get("cat") == "user_annotation":
            tags[e["tid"], e["name"].split("/")[-1]].append(
                (e["ts"], e["ts"] + e["dur"])
            )
    spans = {k: sorted(v) for k, v in tags.items()}
    starts = {k: [lo for lo, hi in v] for k, v in spans.items()}
    labels = (
        "hessian_gate_up",
        "hessian_other",
        "imatrix",
        "model_linear",
        "forward",
        "sample",
        "wait_next_batch",
    )

    def owner(e):
        thread = e["tid"]
        for name in labels:
            key = (thread, name)
            if key in spans:
                i = bisect.bisect_right(starts[key], e["ts"]) - 1
                if i >= 0 and e["ts"] < spans[key][i][1]:
                    return name
        return "unscoped"

    runtimes = {}
    cpu_ops = {}
    waits = Counter()
    copy_api = Counter()
    for e in events:
        cat = e.get("cat")
        args = e.get("args", {})
        if cat in ("cuda_runtime", "cuda_driver"):
            if "correlation" in args:
                runtimes[args["correlation"]] = owner(e)
            if owner(e) != "unscoped":
                if "Synchroniz" in e["name"]:
                    waits[e["name"]] += e["dur"] / 1000
                elif "Memcpy" in e["name"]:
                    copy_api[e["name"]] += e["dur"] / 1000
        if cat == "cpu_op" and "External id" in args:
            cpu_ops[args["External id"]] = e
    gpu = defaultdict(list)
    kernels = Counter()
    copies = Counter()
    copy_bytes = Counter()
    gpu_intervals = []
    forward_spans = [
        s
        for (_, name), intervals in spans.items()
        if name == "forward"
        for s in intervals
    ]
    for e in events:
        cat = e.get("cat")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        args = e.get("args", {})
        label = runtimes.get(args.get("correlation"))
        if label is None or label == "unscoped":
            op = cpu_ops.get(args.get("External id"))
            label = owner(op) if op else "unscoped"
        lo, hi = e["ts"], e["ts"] + e["dur"]
        gpu[label].append((lo, hi))
        gpu_intervals.append((lo, hi))
        if cat == "kernel":
            kernels[label, e["name"]] += e["dur"] / 1000
        else:
            copies[e["name"]] += e["dur"] / 1000
            copy_bytes[e["name"]] += args.get("bytes", 0)
    if (
        len(forward_spans) != 3
        or not gpu.get("hessian_gate_up")
        or not gpu.get("imatrix")
    ):
        raise ValueError(
            f"Missing profiler correlation or forward scopes: {list(gpu)}, {len(forward_spans)}"
        )
    fw_wall = union_us(forward_spans)
    fw_active = union_us(
        (max(l, a), min(h, b))
        for l, h in forward_spans
        for a, b in gpu_intervals
        if a < h and b > l
    )
    return {
        "file": path.name,
        "forward_cpu_wall_ms": fw_wall / 1000,
        "gpu_busy_within_forward_wall_ms": fw_active / 1000,
        "gpu_idle_within_forward_wall_ms": (fw_wall - fw_active) / 1000,
        "cpu_range_inclusive_ms": {
            n: sum(union_us(v) for (_, tag), v in spans.items() if tag == n) / 1000
            for n in labels
        },
        "gpu_active_union_ms_by_submitter": {
            k: union_us(v) / 1000 for k, v in gpu.items()
        },
        "gpu_activity_counts_by_submitter": {k: len(v) for k, v in gpu.items()},
        "device_copy_ms": dict(copies),
        "device_copy_bytes": dict(copy_bytes),
        "sync_api_host_ms_within_markers": dict(waits),
        "copy_api_host_ms_within_markers": dict(copy_api),
        "top_kernels_by_sum_ms": [
            {"range": k, "kernel": n, "ms": v} for (k, n), v in kernels.most_common(9)
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    out = [summarize(a.trace_dir / f"window-{i}.json") for i in (8, 40)]
    a.output.write_text(json.dumps(out, indent=2) + "\n")
