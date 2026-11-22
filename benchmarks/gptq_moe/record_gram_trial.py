"""Verify complete checkpoint byte parity and reduce the separate Gram A/B runs."""

import argparse
import hashlib
import json
from pathlib import Path

from record_forward_results import device, read, timing


def hashes(path):
    result = {}
    for item in sorted(path.iterdir()):
        if item.is_file():
            with item.open("rb") as f:
                result[item.name] = hashlib.file_digest(f, "sha256").hexdigest()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    root = a.raw / "reuse"
    validations = {}
    for name, stock, trial in [
        ("granite_on", a.raw / "granite/checkpoint-on", root / "checkpoint"),
        ("granite_off", a.raw / "granite/checkpoint-off", root / "off-checkpoint"),
        ("tinymoe_on", a.raw / "tinymoe/checkpoint-on", root / "tiny-checkpoint"),
    ]:
        original, candidate = hashes(stock), hashes(trial)
        if "model.safetensors" not in original or original != candidate:
            raise ValueError(f"Checkpoint byte mismatch: {name}")
        validations[name] = {"all_files_identical": True, "sha256": original}
    trials = []
    for i in range(1, 5):
        row = {"order": ["reuse", "base"] if i % 2 else ["base", "reuse"]}
        for arm in row["order"]:
            stem = f"r{i}-{arm}"
            row[arm] = {
                **timing(root / f"{stem}.json"),
                "device": device(root / f"{stem}.device.json"),
            }
        row["calls"] = read(root / f"r{i}-reuse.reuse.json")
        row["first_saved_s"] = row["base"]["first_s"] - row["reuse"]["first_s"]
        row["wall_saved_s"] = row["base"]["oneshot_s"] - row["reuse"]["oneshot_s"]
        row["wall_saved_percent"] = 100 * row["wall_saved_s"] / row["base"]["oneshot_s"]
        trials.append(row)
    out = {
        "pairs": trials,
        "parity": validations,
        "extra_counts": {
            k: read(root / f"{k}.reuse.json") for k in ("tiny-on", "granite-off")
        },
        "extra_devices": {
            k: device(root / f"{k}.device.json")
            for k in ("test-final", "tiny-on", "granite-off")
        },
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n")
