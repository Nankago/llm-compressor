"""Run one diagnostic Python command under the benchmark's NVML idle-card guard."""

import argparse
import os
import sys
from pathlib import Path

from observer_trials import HERE, cache_env, launch, nvml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument(
        "--prefix",
        type=Path,
        required=True,
        help="Prefix for server log and device JSON",
    )
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--kl-tools", type=Path)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.command:
        p.error("Pass a Python script and its arguments")
    a.prefix.parent.mkdir(parents=True, exist_ok=True)
    env = cache_env(a.cache_root, HERE.parents[1] / "src", a.gpu)
    if a.kl_tools:
        env["PYTHONPATH"] += os.pathsep + str(a.kl_tools.resolve())
    nvml.nvmlInit()
    launch(
        [sys.executable, *a.command],
        env,
        nvml.nvmlDeviceGetHandleByUUID(a.gpu),
        a.prefix,
    )


if __name__ == "__main__":
    main()
