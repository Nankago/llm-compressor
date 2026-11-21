"""Normalize per-run JSON while preserving every measured value and batch."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = {}
    layouts = {}
    env = None
    for path in sorted(args.input_dir.glob("*.json")):
        run = json.loads(path.read_text())
        conditions = run.pop("environment")
        if env is None:
            env = conditions
        assert conditions == env
        batches = run.pop("batches")
        rows = "".join(
            json.dumps(batch, separators=(",", ":")) + "\n" for batch in batches
        )
        digest = hashlib.sha256(rows.encode()).hexdigest()[:12]
        name = f"batches-{run['mode'] or 'none'}-{digest}.jsonl"
        if name in layouts:
            assert rows == layouts[name]
        layouts[name] = rows
        run["batch_records"] = name
        raw[path.stem] = run
    if not raw:
        raise ValueError("No replicate JSON files")
    for name, rows in layouts.items():
        (args.output_dir / name).write_text(rows)
    for name, record in [("environment", env), ("runs", raw)]:
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )

    formal = [run for label, run in raw.items() if label[:2].isdigit()]
    for field in (
        "upstream",
        "revision",
        "weight_sha256",
        "seed",
        "sequence_length",
        "samples",
        "input_ids_sha256_le_i64",
    ):
        assert len({run[field] for run in formal}) == 1, field
    stages = list(formal[0]["cuda_event_s"])
    assert all(set(run["cuda_event_s"]) == set(stages) for run in formal)
    with (args.output_dir / "summary.csv").open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(
            [
                "mode",
                "repetitions",
                "batch_size:count",
                "per_item_modules",
                "batched_modules",
                *stages,
                "unattributed_wall_s",
                "oneshot_wall_s",
                "peak_allocated_MiB",
                "peak_reserved_MiB",
            ]
        )
        for mode in ("auto", None):
            runs = [
                r
                for label, r in raw.items()
                if label[:2].isdigit() and r["mode"] == mode
            ]
            assert len(runs) == 3
            first = runs[0]
            assert all(r["batch_records"] == first["batch_records"] for r in runs)
            summary = first["module_count_by_cholesky"]
            sizes = ",".join(
                f"{k}:{v}"
                for k, v in sorted(
                    first["batch_histogram"].items(), key=lambda item: int(item[0])
                )
            )
            values = [median(r["cuda_event_s"][key] for r in runs) for key in stages]
            values += [
                median(r[key] for r in runs)
                for key in ("unattributed_wall_s", "oneshot_wall_s")
            ]
            values += [
                median(r[key] for r in runs) / 2**20
                for key in ("peak_allocated_bytes", "peak_reserved_bytes")
            ]
            writer.writerow(
                [
                    mode or "None",
                    len(runs),
                    sizes,
                    summary.get("per_item", 0),
                    summary.get("batched", 0),
                    *(round(v, 6) for v in values),
                ]
            )


if __name__ == "__main__":
    main()
