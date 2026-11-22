"""Repeat the published oneshot with optional low-overhead scale-search probes."""

import argparse
import cProfile
import gc
import hashlib
import json
import os
import pstats
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import run
from llmcompressor.core import reset_session
from llmcompressor.observers import grid_search, imatrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=run.CHECKPOINTS, default="tinymoe")
    parser.add_argument("--output-digests", action="store_true")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--split", action="store_true")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Diagnostic first-search cProfile and later qparam timers",
    )
    args = parser.parse_args()
    impl = Path(grid_search.__file__).resolve()
    if source := os.getenv("OBSERVER_EXPECT_SRC"):
        assert impl.is_relative_to(Path(source).resolve()), impl
    args.output.mkdir(parents=True, exist_ok=False)
    for index in range(args.passes):
        reset_session()
        gc.collect()
        torch.cuda.empty_cache()
        raw = args.output / f"pass{index+1}.json"
        spans = []
        original = imatrix._grid_search_observer
        profile = cProfile.Profile() if args.diagnose else None
        qparam_ms = []
        calculate = grid_search.calculate_qparams

        def measured_qparams(*values, **kwargs):
            t0 = time.perf_counter()
            result = calculate(*values, **kwargs)
            qparam_ms.append((time.perf_counter() - t0) * 1000)
            return result

        def measured(*values, **kwargs):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            t0 = time.perf_counter()
            if profile is not None and not spans:
                result = profile.runcall(original, *values, **kwargs)
            else:
                result = original(*values, **kwargs)
            elapsed = time.perf_counter() - t0
            end.record()
            spans.append((start, end, elapsed))
            return result

        with ExitStack() as stack:
            if args.diagnose:
                stack.enter_context(
                    patch.object(grid_search, "calculate_qparams", measured_qparams)
                )
            if args.split or args.diagnose:
                stack.enter_context(
                    patch.object(imatrix, "_grid_search_observer", measured)
                )
            stack.enter_context(
                patch.object(
                    sys,
                    "argv",
                    [
                        "run.py",
                        "--mode",
                        "auto",
                        "--samples",
                        "8",
                        "--observer-details",
                        "--model-dir",
                        str(args.model_dir),
                        "--out",
                        str(raw),
                        "--checkpoint",
                        args.checkpoint,
                        *(["--output-digests"] if args.output_digests else []),
                    ],
                )
            )
            run.main()
        data = json.loads(raw.read_text())
        data["grid_search_source_sha256"] = hashlib.sha256(
            impl.read_bytes()
        ).hexdigest()
        if profile is not None:
            stats = pstats.Stats(profile)
            rows = []
            for (file, line, fn), (_, count, own, cumulative, _) in stats.stats.items():
                visible = "/".join(Path(file).parts[-3:]) if "/" in file else file
                rows.append(
                    {
                        "function": f"{visible}:{line} {fn}",
                        "calls": count,
                        "self_s": own,
                        "inclusive_s": cumulative,
                    }
                )
            data["first_search_profile"] = sorted(
                rows, key=lambda r: -r["inclusive_s"]
            )[:60]
            data["later_calculate_qparams_host_ms"] = sum(qparam_ms[1:])
            data["calculate_qparams_calls"] = len(qparam_ms)
        if args.split or args.diagnose:
            assert len(spans) == sum(data["module_count_by_cholesky"].values())
            data["scale_search"] = {
                "event_ms": [a.elapsed_time(b) for a, b, _ in spans],
                "host_ms": [s * 1000 for _, _, s in spans],
            }
        raw.write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
