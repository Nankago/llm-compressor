"""Alternate fresh-checkpoint propagation trials on one idle device."""

import argparse
import sys
from pathlib import Path

from observer_trials import HERE, cache_env, launch, nvml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", choices=["granite", "tinymoe"], required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path)
    p.add_argument("--gpu", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--length", type=int, default=1024)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    nvml.nvmlInit()
    device = nvml.nvmlDeviceGetHandleByUUID(a.gpu)
    env = cache_env(a.out / "compiled", HERE.parents[1] / "src", a.gpu)
    Path(env["CUDA_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    for rep in range(a.pairs + 1):
        for arm in (("on", "off") if rep % 2 else ("off", "on")):
            prefix = a.out / f"r{rep}-{arm}"
            cmd = [
                sys.executable,
                str(HERE / "run.py"),
                "--checkpoint",
                a.checkpoint,
                "--model-dir",
                str(a.model_dir),
                "--mode",
                "auto",
                "--coarse-timing",
                "--samples",
                str(a.samples),
                "--seq-length",
                str(a.length),
                "--out",
                str(prefix.with_suffix(".json")),
            ]
            if a.calibration_dir:
                cmd += ["--calibration-dir", str(a.calibration_dir)]
            if arm == "off":
                cmd += ["--no-propagate-error"]
            if rep == 1:
                cmd += ["--save-checkpoint", str(a.out / f"checkpoint-{arm}")]
            launch(cmd, env, device, prefix)
            print(prefix.name, flush=True)


if __name__ == "__main__":
    main()
