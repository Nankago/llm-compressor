"""Compact raw processes and verify source, checkpoint, calibration and output identity."""

import argparse
import json
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def compact(root):
    summary = {
        "models": {},
        "runs": [],
        "device_records": {},
        "trace": {},
        "first_search_profiles": {},
        "source_sha256": {},
    }
    for series in ("matrix", "diagnosis", "tiny-ab", "granite-ab", "extras"):
        leaf = root / series
        for f in sorted(leaf.glob("**/pass*.json")):
            run = read(f)
            rel = f.parent.relative_to(root).as_posix()
            device = read(f.parent.with_suffix(".device.json"))
            assert device["returncode"] == 0 and not device["foreign_pids"]
            assert not device["before"]["pids"] and not device["after"]["pids"]
            assert device["before"]["util_pct"] == 0
            summary["device_records"][rel] = device
            if "environment" not in summary:
                summary["environment"] = run["environment"]
                summary["gpu"] = run["gpu_before"]
            assert run["environment"] == summary["environment"]
            model = run["model"]
            identity = {
                k: run[k]
                for k in (
                    "revision",
                    "weight_sha256",
                    "seed",
                    "samples",
                    "sequence_length",
                    "token_offsets",
                    "input_ids_sha256_le_i64",
                    "mode",
                    "batch_histogram",
                    "module_count_by_cholesky",
                    "rtn_fallback_modules",
                )
            }
            entry = summary["models"].setdefault(model, {"workload": identity})
            assert entry["workload"] == identity
            assert not run["rtn_fallback_modules"] and run["mode"] == "auto"
            if "output_digests" in run:
                outputs = run["output_digests"]
                assert {"weight", "weight_scale"} <= set(outputs)
                assert all(
                    x["tensors"] == sum(run["module_count_by_cholesky"].values())
                    for x in outputs.values()
                )
                assert entry.setdefault("output_digests", outputs) == outputs
                label = f.parent.name.split("-")[-1]
                hashes = summary["source_sha256"]
                assert (
                    hashes.setdefault(label, run["grid_search_source_sha256"])
                    == run["grid_search_source_sha256"]
                )
            row = {
                "process": rel,
                "pass": int(f.stem[4:]),
                "start": run["start_utc"],
                "oneshot_s": run["oneshot_wall_s"],
                "observer_s": run["cuda_event_s"]["weight_observe"],
                "peak_allocated_bytes": run["peak_allocated_bytes"],
                "peak_reserved_bytes": run["peak_reserved_bytes"],
            }
            if "scale_search" in run:
                for timer, spans in run["scale_search"].items():
                    row["search_calls"] = len(spans)
                    row["first_" + timer] = spans[0]
                    row["later_" + timer] = sum(spans[1:])
            if series == "diagnosis":
                row["later_calculate_qparams_host_ms"] = run[
                    "later_calculate_qparams_host_ms"
                ]
                summary["first_search_profiles"][rel + "/" + f.stem] = run[
                    "first_search_profile"
                ]
            summary["runs"].append(row)
    for name in ("tiny-ab", "granite-ab"):
        assert (
            sum(row["process"].startswith(name + "/") for row in summary["runs"]) == 18
        )
    assert sum(row["process"].startswith("matrix/") for row in summary["runs"]) == 18
    assert summary["source_sha256"]["base"] != summary["source_sha256"]["proto"]
    for label in ("base", "proto"):
        prefix = root / "extras" / ("warm-" + label)
        summary["trace"][label] = {
            "server_file": str(prefix.with_suffix(".nsys-rep")),
            "device": read(prefix.with_suffix(".device.json")),
            "ranges": read(root / "extras" / (label + "-trace-summary.json")),
        }
    edges = [
        read(root / "extras" / (label + "-edge.json")) for label in ("base", "proto")
    ]
    assert edges[0] == edges[1] and len(edges[0]) == 16
    summary["edge_digests"] = edges[0]
    for name in ("tests", "base-edge", "proto-edge"):
        device = read(root / "extras" / (name + ".device.json"))
        assert device["returncode"] == 0 and not device["foreign_pids"]
        summary["device_records"]["extras/" + name] = device
    lines = (root / "extras/tests.log").read_text().splitlines()
    summary["test_result"] = next(
        s for s in reversed(lines) if re.search(r"\d+ passed", s)
    )
    assert sum(row["process"].startswith("extras/") for row in summary["runs"]) == 4
    for checkpoint in ("tinymoe", "granite"):
        for label in ("base", "proto"):
            prefix = root / "extras" / f"packed-{checkpoint}-{label}"
            device = read(prefix.with_suffix(".device.json"))
            assert device["returncode"] == 0 and not device["foreign_pids"]
            summary["device_records"][str(prefix.relative_to(root))] = device
            run = read(prefix.with_suffix(".json"))
            entry = summary["models"][run["model"]]
            packed = run["packed_digests"]
            assert packed["weight_scale"] == entry["output_digests"]["weight_scale"]
            assert (
                packed["weight_packed"]["tensors"]
                == entry["output_digests"]["weight"]["tensors"]
            )
            assert entry.setdefault("packed_digests", packed) == packed
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(compact(args.raw.resolve()), indent=2) + "\n")
