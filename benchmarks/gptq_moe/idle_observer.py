"""Select a quiescent H100, run three reference replicates, then capture one trace."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml as nvml

HERE = Path(__file__).resolve().parent
INPUT_SHA = "b3d5839fee2d2adc20c570fe826ffd556a8f4a37c5b81fe0c9720681d4b400a0"


def inventory():
    devices = []
    for index in range(nvml.nvmlDeviceGetCount()):
        gpu = nvml.nvmlDeviceGetHandleByIndex(index)
        name = nvml.nvmlDeviceGetName(gpu)
        if "H100" not in name:
            continue
        processes = []
        for kind, query in [
            ("compute", nvml.nvmlDeviceGetComputeRunningProcesses),
            ("graphics", nvml.nvmlDeviceGetGraphicsRunningProcesses),
        ]:
            for proc in query(gpu):
                try:
                    pname = Path(nvml.nvmlSystemGetProcessName(proc.pid)).name
                except nvml.NVMLError_NotFound:
                    pname = "exited"
                processes.append(
                    {
                        "type": kind,
                        "pid": proc.pid,
                        "name": pname,
                        "used_MiB": (
                            proc.usedGpuMemory // 2**20
                            if proc.usedGpuMemory is not None
                            else None
                        ),
                    }
                )
        mem = nvml.nvmlDeviceGetMemoryInfo(gpu)
        devices.append(
            {
                "index": index,
                "uuid": nvml.nvmlDeviceGetUUID(gpu),
                "name": name,
                "util_pct": nvml.nvmlDeviceGetUtilizationRates(gpu).gpu,
                "used_MiB": mem.used // 2**20,
                "total_MiB": mem.total // 2**20,
                "processes": processes,
            }
        )
    return {"utc": datetime.now(timezone.utc).isoformat(), "devices": devices}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def idle_ids(snapshot):
    return {
        d["uuid"]
        for d in snapshot["devices"]
        if not d["processes"] and d["util_pct"] == 0
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="New artifact directory outside Git"
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--wait-hourly",
        action="store_true",
        help="When occupied, sleep one hour before checking again",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    nvml.nvmlInit()
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    nsys_version = subprocess.check_output(["nsys", "--version"], text=True).strip()
    while True:
        checks = [inventory()]
        available = idle_ids(checks[0])
        # Busy checks are one NVML sample per hour. Only confirm apparent vacancies.
        for _ in range(2):
            if not available:
                break
            time.sleep(2)
            checks.append(inventory())
            available &= idle_ids(checks[-1])
        selected = next(
            (d["uuid"] for d in checks[-1]["devices"] if d["uuid"] in available), None
        )
        status = {
            "base_commit": base_commit,
            "checks": checks,
            "selected_gpu": selected,
            "status": "ready" if selected else "no_idle_h100",
            "nsys": nsys_version,
        }
        save(out / "preflight.json", status)
        with (out / "availability.jsonl").open("a") as log:
            log.write(json.dumps(status) + "\n")
        if selected:
            break
        print(checks[-1]["utc"], "No idle H100", flush=True)
        if not args.wait_hourly:
            return 2
        time.sleep(3600)

    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=selected,
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TRITON_CACHE_DIR=str(out / "triton"),
    )
    results = []
    report = out / "observer-trace"
    for name in ("warmup", "run1", "run2", "run3", "trace"):
        before = inventory()
        # NVML utilization can still cover the last process after it has exited.
        for _ in range(4):
            device = next(d for d in before["devices"] if d["uuid"] == selected)
            if device["processes"] or device["util_pct"] == 0:
                break
            time.sleep(1)
            before = inventory()
        save(out / f"{name}-before.json", before)
        if selected not in idle_ids(before):
            raise RuntimeError(f"{selected} is busy before {name}")
        entry = HERE / ("observer_nvtx.py" if name == "trace" else "run.py")
        cmd = [
            sys.executable,
            str(entry),
            "--mode",
            "auto",
            "--samples",
            "8",
            "--observer-details",
            "--out",
            str(out / f"{name}.json"),
        ]
        if args.model_dir:
            cmd.extend(["--model-dir", str(args.model_dir.resolve())])
        if name == "trace":
            cmd = [
                "nsys",
                "profile",
                "--trace=cuda,nvtx,osrt,python-gil",
                "--cuda-trace-all-apis=true",
                "--cuda-trace-scope=process-tree",
                "--sample=process-tree",
                "--cpuctxsw=process-tree",
                "--python-sampling=true",
                "--pytorch=autograd-nvtx",
                "-o",
                str(report),
                *cmd,
            ]
        with (out / f"{name}.log").open("w") as log:
            run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        after = inventory()
        save(out / f"{name}-after.json", after)
        run.check_returncode()
        if next(d["processes"] for d in after["devices"] if d["uuid"] == selected):
            raise RuntimeError(f"Another process holds {selected} after {name}")
        record = json.loads((out / f"{name}.json").read_text())
        if (
            record["input_ids_sha256_le_i64"] != INPUT_SHA
            or record["calls"]["weight_observe"] != 11
        ):
            raise ValueError("Calibration or observer coverage changed")
        if name.startswith("run"):
            results.append(
                {
                    "run": name,
                    "oneshot_s": record["oneshot_wall_s"],
                    "weight_observe_s": record["cuda_event_s"]["weight_observe"],
                }
            )
            summary = {}
            for key in ("oneshot_s", "weight_observe_s"):
                times = [row[key] for row in results]
                summary[key] = {
                    "median": statistics.median(times),
                    "range": [min(times), max(times)],
                }
            save(out / "timings.json", {"runs": results, "summary": summary})
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "-o",
            str(report.with_suffix(".sqlite")),
            str(report.with_suffix(".nsys-rep")),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
