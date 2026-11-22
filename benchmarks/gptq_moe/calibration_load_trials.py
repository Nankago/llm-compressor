"""Counterbalance fresh-process Granite baselines and prototypes on fixed WikiText tokens."""

import argparse
import sys
from pathlib import Path

from observer_trials import HERE, cache_env, launch, nvml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-src", type=Path, required=True)
    p.add_argument("--proto-src", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    nvml.nvmlInit()
    dev = nvml.nvmlDeviceGetHandleByUUID(args.gpu)
    sources = {"base": args.base_src, "proto": args.proto_src}
    for workload, n, length in (("heavy", 64, 1024), ("short", 8, 128)):
        root = out / workload
        root.mkdir()
        # Prime both arms on this exact workload. Each timing starts a fresh
        # process and reloads original weights; cache contents are private to its arm.
        for rep in range(5):
            for arm in (("base", "proto") if rep % 2 else ("proto", "base")):
                prefix = root / f"r{rep}-{arm}"
                env = cache_env(root / arm, sources[arm], args.gpu)
                env.pop("TRITON_LIBCUDA_PATH", None)
                Path(env["CUDA_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
                launch(
                    [
                        sys.executable,
                        str(HERE / "run.py"),
                        "--mode",
                        "auto",
                        "--checkpoint",
                        "granite",
                        "--samples",
                        str(n),
                        "--seq-length",
                        str(length),
                        "--calibration-dir",
                        str(args.calibration_dir),
                        "--model-dir",
                        str(args.model_dir),
                        "--coarse-timing",
                        "--output-digests",
                        "--packed-digests",
                        "--out",
                        str(prefix.with_suffix(".json")),
                    ],
                    env,
                    dev,
                    prefix,
                )
                print(workload, rep, arm, "ok", flush=True)


if __name__ == "__main__":
    main()
