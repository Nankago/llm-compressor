"""Summarize paired language-model and task scores."""

import json
import math
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def main():
    root, out = map(Path, sys.argv[1:])
    panels = {
        k: read(root / (k + "-panel.json"))
        for k in ("short", "long", "rep", "native", "long-float", "rep-float")
    }
    reference = panels["short"]
    builds, calibrations, quality, paired = {}, {}, {}, {}
    for name, panel in panels.items():
        assert (
            panel["test"] == reference["test"]
            and panel["positions"] == reference["positions"]
        )
        assert panel["quality"]["bf16"] == reference["quality"]["bf16"]

        def label(k):
            return ("native-" + k) if name == "native" else k

        for k, b in panel["builds"].items():
            cal = b["train"]
            calibrations[cal["sha256"]] = cal
            builds[label(k)] = {
                field: value
                for field, value in b.items()
                if field not in ("train", "layer_seconds")
            }
            builds[label(k)]["calibration_sha256"] = cal["sha256"]
            quality[label(k)] = panel["quality"][k]
        for n, pair in panel["pairs"].items():
            paired[label(n)] = {**pair, "control": label(pair["control"])}
    task = {}
    for key in (
        "bf16",
        "long-stock",
        "long-float",
        "long-sweep",
        "long-gptaq",
        "rep-stock",
        "rep-float",
        "rep-sweep",
        "rep-gptaq",
    ):
        a = read(root / ("arc-" + key + ".json"))
        assert a["row_ids"] == list(range(2376)) and a["evaluated"] == 2376
        if key != "bf16":
            assert a["model_sha256"] == builds[key]["checkpoint_sha256"]
        arr = np.unpackbits(
            np.frombuffer(bytes.fromhex(a["correct_bits"]), dtype=np.uint8),
            bitorder="little",
        )[:2376]
        assert int(arr.sum()) == a["correct"]
        task[key] = {
            k: a[k] for k in ("correct", "correct_bits", "elapsed_s", "model_sha256")
        }
        task[key]["accuracy"] = a["correct"] / 2376
        if key != "bf16" and not key.endswith("-stock"):
            prefix, method = key.split("-")
            comparisons = [prefix + "-stock"]
            if method == "gptaq":
                comparisons.append(prefix + "-float")
            task[key]["paired"] = {}
            for sibling in comparisons:
                bits = bytes.fromhex(task[sibling]["correct_bits"])
                base = np.unpackbits(
                    np.frombuffer(bits, dtype=np.uint8), bitorder="little"
                )[:2376]
                wins = int(np.sum((arr == 1) & (base == 0)))
                losses = int(np.sum((arr == 0) & (base == 1)))
                n = wins + losses
                p = min(
                    1.0,
                    2
                    * sum(math.comb(n, j) for j in range(min(wins, losses) + 1))
                    / 2**n,
                )
                task[key]["paired"][sibling] = dict(
                    wins=wins, losses=losses, mcnemar_exact_p=p
                )
    rechecks = {}
    for key in ("long-gptaq", "rep-gptaq"):
        check = read(root / ("recheck-" + key + ".json"))
        rows = check["row_ids"]
        assert len(rows) == len(set(rows)) == check["evaluated"] == 256
        assert check["model_sha256"] == task[key]["model_sha256"]
        again, original = (
            np.unpackbits(
                np.frombuffer(bytes.fromhex(bits), dtype=np.uint8), bitorder="little"
            )
            for bits in (check["correct_bits"], task[key]["correct_bits"])
        )
        previous = original[rows]
        assert int(again.sum()) == check["correct"]
        rechecks[key] = dict(
            row_ids=rows,
            rerun_correct_bits=check["correct_bits"],
            baseline_correct=int(previous.sum()),
            rerun_correct=int(again.sum()),
            changed_row_ids=np.asarray(rows)[again != previous].tolist(),
        )
    for prefix in ("long", "rep"):
        fitted, control = prefix + "-gptaq", prefix + "-float"
        assert (
            builds[fitted]["calibration_sha256"]
            == builds[control]["calibration_sha256"]
        )
        diff = (
            np.array(quality[fitted]["article_kl_nll"])
            - quality[control]["article_kl_nll"]
        )
        ids = np.random.default_rng(20261126).integers(
            len(diff), size=(10000, len(diff))
        )
        paired[fitted + "-vs-float"] = dict(
            control=control,
            delta_kl=diff[:, 0].mean(),
            delta_nll=diff[:, 1].mean(),
            ci95_kl_nll=np.quantile(diff[ids].mean(1), [0.025, 0.975], axis=0).tolist(),
        )
    data = dict(
        base="18b10e2d7269e332c05eef4dda52adf74c08872e",
        environment=dict(
            cpu=next(
                line.partition(":")[2].strip()
                for line in subprocess.check_output(["lscpu"], text=True).splitlines()
                if line.startswith("Model name:")
            ),
            threads=torch.get_num_threads(),
            python=platform.python_version(),
            packages={
                k: version(k)
                for k in (
                    "torch",
                    "transformers",
                    "compressed-tensors",
                    "safetensors",
                    "pyarrow",
                    "numpy",
                )
            },
        ),
        calibration=calibrations,
        test=reference["test"],
        scored_positions=reference["positions"],
        bf16=reference["quality"]["bf16"],
        builds=builds,
        quality=quality,
        paired=paired,
        arc_easy=task,
        arc_subset_recheck=rechecks,
        oneshot_first_block=read(root / "native/native.json"),
        oneshot_full=read(root / "native-full/native.json"),
    )
    qda = root / "qda-check.json"
    if qda.exists():
        data["qda_reference"] = read(qda)
    out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
