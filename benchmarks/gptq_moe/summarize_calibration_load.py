"""Verify and summarize the paired calibration-load runs without storing model data."""

import argparse
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    return {"median": median(values), "min": min(values), "max": max(values)}


def summarize(root):
    workloads = {}
    result = {"workloads": workloads, "observer_source_sha256": {}}
    for load, shape in (("short", (8, 128)), ("heavy", (64, 1024))):
        trials = []
        target = workloads[load] = {"runs": trials}
        for rep in range(5):
            for arm in (("base", "proto") if rep % 2 else ("proto", "base")):
                prefix = root / load / f"r{rep}-{arm}"
                d = read(prefix.with_suffix(".json"))
                device = read(prefix.with_suffix(".device.json"))
                assert device["returncode"] == 0 and not device["foreign_pids"]
                assert not device["before"]["pids"] and not device["after"]["pids"]
                assert device["before"]["util_pct"] == 0
                assert (d["samples"], d["sequence_length"]) == shape
                assert (
                    not d["rtn_fallback_modules"]
                    and d["mode"] == "auto"
                    and d["timing"] == "coarse"
                )
                assert d["calls"]["weight_observe"] == 25
                assert d["calls"]["calib_forward_including_hooks"] == shape[0] * 25
                assert d["module_count_by_cholesky"] == {
                    "per_item": 48,
                    "batched": 2352,
                }
                expected_fields = {"weight", "weight_scale", "weight_zero_point"}
                assert set(d["output_digests"]) == expected_fields
                assert set(d["packed_digests"]) == {
                    "weight_packed",
                    "weight_scale",
                    "weight_shape",
                }
                for digest in [
                    *d["packed_digests"].values(),
                    *d["output_digests"].values(),
                ]:
                    assert digest["tensors"] == 2400
                assert (
                    d["packed_digests"]["weight_scale"]
                    == d["output_digests"]["weight_scale"]
                )
                for field in (
                    "model",
                    "revision",
                    "weight_sha256",
                    "upstream",
                    "environment",
                    "module_count_by_cholesky",
                    "batch_histogram",
                    "calibration_bank",
                ):
                    assert result.setdefault(field, d[field]) == d[field]
                # The full-run calibration data and every tensor hash must match between arms.
                for field in (
                    "samples",
                    "sequence_length",
                    "input_ids_sha256_le_i64",
                    "token_offsets",
                    "output_digests",
                    "packed_digests",
                ):
                    assert target.setdefault(field, d[field]) == d[field]
                assert (
                    result["observer_source_sha256"].setdefault(
                        arm, d["observer_source_sha256"]
                    )
                    == d["observer_source_sha256"]
                )
                gpu = d["gpu_before"][1].split(", ")[:5]
                assert result.setdefault("device", gpu) == gpu
                stages = d["cuda_event_s"]
                trials.append(
                    {
                        "round": rep,
                        "arm": arm,
                        "prime": rep == 0,
                        "observer_s": stages["weight_observe"],
                        "oneshot_s": d["oneshot_wall_s"],
                        "calib_forward_s": stages["calib_forward_including_hooks"],
                        "propagate_forward_s": stages["propagate_forward"],
                        "peak_allocated_mib": d["peak_allocated_bytes"] / 2**20,
                        "peak_reserved_mib": d["peak_reserved_bytes"] / 2**20,
                        "device_record": device,
                    }
                )

        measured = {
            arm: [row for row in trials if row["arm"] == arm and not row["prime"]]
            for arm in ("base", "proto")
        }
        target["arms"] = {
            arm: {
                k: distribution([r[k] for r in rows])
                for k in (
                    "observer_s",
                    "oneshot_s",
                    "calib_forward_s",
                    "propagate_forward_s",
                    "peak_allocated_mib",
                    "peak_reserved_mib",
                )
            }
            for arm, rows in measured.items()
        }
        pairs = []
        for a, b in zip(measured["base"], measured["proto"], strict=True):
            assert a["round"] == b["round"]
            wall = a["oneshot_s"] - b["oneshot_s"]
            obs = a["observer_s"] - b["observer_s"]
            pairs.append(
                {
                    "round": a["round"],
                    "observer_saved_s": obs,
                    "oneshot_saved_s": wall,
                    "oneshot_saved_pct": 100 * wall / a["oneshot_s"],
                    "observer_saved_pct_of_baseline_wall": 100 * obs / a["oneshot_s"],
                    "nonobserver_wall_delta_s": wall - obs,
                }
            )
        target["pairs"] = pairs
        target["paired_summary"] = {
            k: distribution([row[k] for row in pairs]) for k in pairs[0] if k != "round"
        }
    assert (
        result["observer_source_sha256"]["base"]
        != result["observer_source_sha256"]["proto"]
    )
    assert (
        workloads["heavy"]["input_ids_sha256_le_i64"]
        == result["calibration_bank"]["input_ids_sha256_le_i64"]
    )
    assert (
        workloads["short"]["token_offsets"] == workloads["heavy"]["token_offsets"][:8]
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.trials), indent=2) + "\n")
