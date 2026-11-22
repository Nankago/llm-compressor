"""Collect independent diagnostic profiles, warmed traces and parity checks."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from observer_trials import HERE, cache_env, launch, nvml, sample


def capture(env, dev, label, dest, model):
    prefix = dest / ("warm-" + label)
    before = sample(dev)
    if before["pids"] or before["util_pct"]:
        raise RuntimeError(f"Busy before diagnostic trace: {before}")
    cmd = [
        "nsys",
        "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--trace=cuda,nvtx",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "-o",
        str(prefix),
        sys.executable,
        str(HERE / "warm_search_trace.py"),
        "--mode",
        "auto",
        "--samples",
        "8",
        "--observer-details",
        "--model-dir",
        str(model),
        "--out",
        str(prefix.with_suffix(".json")),
    ]
    with prefix.with_suffix(".log").open("w") as log:
        subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    after = sample(dev)
    prefix.with_suffix(".device.json").write_text(
        json.dumps({"before": before, "after": after})
    )
    if after["pids"]:
        raise RuntimeError(f"Busy after diagnostic trace: {after}")
    sqlite = prefix.with_suffix(".sqlite")
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "-o",
            str(sqlite),
            str(prefix.with_suffix(".nsys-rep")),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "summarize_observer_trace.py"),
            "--warm-window",
            label,
            str(sqlite),
            str(dest / (label + "-trace-summary.json")),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, required=True, help="Root containing tiny-ab"
    )
    parser.add_argument("--baseline-src", type=Path, required=True)
    parser.add_argument("--prototype-src", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--granite-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument(
        "--libcuda-dir",
        type=Path,
        required=True,
        help="Same driver library directory selected by Triton discovery on this host",
    )
    args = parser.parse_args()
    root = args.raw.resolve()
    model = args.model_dir.resolve()
    nvml.nvmlInit()
    device = nvml.nvmlDeviceGetHandleByUUID(args.gpu)
    profile = root / "diagnosis"
    profile.mkdir(exist_ok=False)
    dest = root / "extras"
    dest.mkdir(exist_ok=False)
    env = cache_env(profile, args.baseline_src, args.gpu)
    for state, passes in (("cold", 1), ("disk", 2)):
        launch(
            [
                sys.executable,
                str(HERE / "observer_pass.py"),
                "--diagnose",
                "--passes",
                str(passes),
                "--model-dir",
                str(model),
                "--output",
                str(profile / state),
            ],
            env,
            device,
            profile / state,
        )

    envs = {
        label: cache_env(root / "tiny-ab" / label, src, args.gpu)
        for label, src in [("base", args.baseline_src), ("proto", args.prototype_src)]
    }
    for label, env in envs.items():
        capture(env, device, label, dest, model)
    if not (args.libcuda_dir / "libcuda.so.1").is_file():
        parser.error("Missing libcuda.so.1 in --libcuda-dir")
    for i, label in enumerate(("lookup", "explicit", "explicit", "lookup")):
        env = envs["base"].copy()
        env.pop("TRITON_LIBCUDA_PATH", None)
        if label == "explicit":
            env["TRITON_LIBCUDA_PATH"] = str(args.libcuda_dir)
        prefix = dest / f"{i}-{label}"
        launch(
            [
                sys.executable,
                str(HERE / "observer_pass.py"),
                "--split",
                "--model-dir",
                str(model),
                "--output",
                str(prefix),
            ],
            env,
            device,
            prefix,
        )
    test_env = dict(envs["proto"], PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    launch(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/llmcompressor/observers/test_imatrix.py",
            "tests/llmcompressor/observers/test_mse.py",
        ],
        test_env,
        device,
        dest / "tests",
    )
    for label, env in envs.items():
        launch(
            [
                sys.executable,
                str(HERE / "check_search_bounds.py"),
                str(dest / (label + "-edge.json")),
            ],
            env,
            device,
            dest / (label + "-edge"),
        )
    for checkpoint, model_dir in (("tinymoe", model), ("granite", args.granite_dir)):
        for label, env in envs.items():
            prefix = dest / f"packed-{checkpoint}-{label}"
            launch(
                [
                    sys.executable,
                    str(HERE / "run.py"),
                    "--packed-digests",
                    "--observer-details",
                    "--checkpoint",
                    checkpoint,
                    "--mode",
                    "auto",
                    "--model-dir",
                    str(model_dir),
                    "--out",
                    str(prefix.with_suffix(".json")),
                ],
                env,
                device,
                prefix,
            )


if __name__ == "__main__":
    main()
