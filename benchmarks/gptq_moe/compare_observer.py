"""Compare two source trees in counterbalanced fresh processes on one idle GPU."""

import argparse
import os
import sys
from pathlib import Path

from observer_trials import HERE, launch, nvml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-src", required=True, type=Path)
    p.add_argument("--prototype-src", required=True, type=Path)
    p.add_argument("--checkpoint", choices=["tinymoe", "granite"], required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=4)
    args = p.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    nvml.nvmlInit()
    device = nvml.nvmlDeviceGetHandleByUUID(args.gpu)
    sources = {"base": args.baseline_src, "proto": args.prototype_src}
    for rep in range(args.repeats + 1):
        for label in (["base", "proto"] if rep % 2 else ["proto", "base"]):
            # Warm each implementation's private disk cache once. Timed processes
            # reuse only their own implementation's already populated cache.
            cache = root / label
            src = sources[label].resolve()
            prefix = root / f"r{rep}-{label}"
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=args.gpu,
                HF_HUB_OFFLINE="1",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                PYTHONPATH=str(src),
                OBSERVER_EXPECT_SRC=str(src),
                TRITON_CACHE_DIR=str(cache / "triton"),
                CUDA_CACHE_PATH=str(cache / "cuda"),
                TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
                TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
            )
            (cache / "cuda").mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(HERE / "observer_pass.py"),
                "--output",
                str(prefix),
                "--model-dir",
                str(args.model_dir),
                "--checkpoint",
                args.checkpoint,
                "--passes",
                "1" if rep == 0 else "2",
                "--output-digests",
            ]
            launch(cmd, env, device, prefix)
            print(args.checkpoint, prefix.name, "ok", flush=True)


if __name__ == "__main__":
    main()
