"""Run independent cold-cache, warm-disk and second-pass trials on one idle GPU."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml as nvml

HERE = Path(__file__).resolve().parent


def cache_env(root, src, gpu):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRITON_CACHE_DIR=str(root / "triton"),
        CUDA_CACHE_PATH=str(root / "cuda"),
        TORCH_EXTENSIONS_DIR=str(root / "extensions"),
        TORCHINDUCTOR_CACHE_DIR=str(root / "inductor"),
    )
    if src is not None:
        env["PYTHONPATH"] = str(src.resolve())
        env["OBSERVER_EXPECT_SRC"] = str(src.resolve())
    return env


def sample(device):
    processes = nvml.nvmlDeviceGetComputeRunningProcesses(
        device
    ) + nvml.nvmlDeviceGetGraphicsRunningProcesses(device)
    mem = nvml.nvmlDeviceGetMemoryInfo(device)
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "util_pct": nvml.nvmlDeviceGetUtilizationRates(device).gpu,
        "used_MiB": mem.used // 2**20,
        "pids": sorted({p.pid for p in processes}),
    }


def launch(cmd, env, device, prefix):
    for _ in range(5):
        before = sample(device)
        if before["pids"] or not before["util_pct"]:
            break
        time.sleep(1)
    if before["pids"] or before["util_pct"]:
        raise RuntimeError(f"GPU busy at entry: {before}")
    foreign = set()
    samples = 0
    with prefix.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            snap = sample(device)
            samples += 1
            foreign.update(p for p in snap["pids"] if p != process.pid)
            if foreign:
                process.terminate()
                process.wait()
                break
            time.sleep(0.5)
    after = sample(device)
    record = {
        "before": before,
        "after": after,
        "foreign_pids": sorted(foreign),
        "samples": samples,
        "returncode": process.returncode,
        "pid": process.pid,
    }
    prefix.with_suffix(".device.json").write_text(json.dumps(record, indent=2) + "\n")
    if process.returncode or foreign or after["pids"]:
        raise RuntimeError(f"Failed or contended trial: {prefix}, {record}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU UUID")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    nvml.nvmlInit()
    dev = nvml.nvmlDeviceGetHandleByUUID(args.gpu)
    for rep in range(1, args.repeats + 1):
        for probe in (["reference", "split"] if rep % 2 else ["split", "reference"]):
            base = out / f"r{rep}-{probe}"
            base.mkdir()
            env = cache_env(base, None, args.gpu)
            for condition, passes in (("cold", 1), ("disk", 2)):
                prefix = base / condition
                cmd = [
                    sys.executable,
                    str(HERE / "observer_pass.py"),
                    "--passes",
                    str(passes),
                    "--output",
                    str(prefix),
                    "--model-dir",
                    str(args.model_dir),
                ]
                if probe == "split":
                    cmd.append("--split")
                launch(cmd, env, dev, prefix)
                print(base.name, condition, "ok", flush=True)


if __name__ == "__main__":
    main()
