"""Profile two short first-pass windows inside a real 64x1024 Granite calibration."""

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import torch
from llmcompressor.modifiers.gptq import base
from llmcompressor.pipelines.sequential import pipeline
from llmcompressor.pipelines.sequential.helpers import Subgraph
from torch.profiler import ProfilerActivity, profile, record_function

import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    original_batches = pipeline._get_batches
    original_forward = Subgraph.forward
    original_h = base.accumulate_hessian
    original_linear = torch.nn.functional.linear
    records = []
    examples = []
    active = False

    def get_batches(cache, n, names, desc):
        nonlocal active
        it = iter(original_batches(cache, n, names, desc))
        if desc != "(13/25): Calibrating":
            yield from it
            return
        i = 0
        while i < n:
            if i not in (8, 40):
                yield next(it)
                i += 1
                continue
            start = i
            active = True
            try:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as prof:
                    for _ in range(3):
                        with record_function("ablation/wait_next_batch"):
                            value = next(it)
                        with record_function("ablation/sample"):
                            yield value
                        i += 1
            finally:
                active = False
            prof.export_chrome_trace(str(a.out / f"window-{start}.json"))
            # Persist a real gate/up input for a later numerical replay, outside capture.
            from safetensors.torch import save_file

            save_file(
                {
                    f"sample_{start+j}": x.detach().contiguous().cpu()
                    for j, x in enumerate(examples)
                },
                a.out / f"replay-{start}.safetensors",
            )
            examples.clear()

    def forward(graph, model, **kwargs):
        if not active:
            return original_forward(graph, model, **kwargs)
        names = {m: n for n, m in graph.submodule_dict(model).items()}
        counts = Counter()
        shared = {}
        refs = (
            []
        )  # keep storages live so allocator recycling cannot fake input equality

        def h(inp, module, *args, **kw):
            name = names[module]
            gateup = ".experts." in name and name.endswith(("gate_proj", "up_proj"))
            group = "gate_up" if gateup else "other"
            if gateup:
                key = (
                    inp.data_ptr(),
                    tuple(inp.shape),
                    tuple(inp.stride()),
                    str(inp.dtype),
                )
                shared.setdefault(key, []).append(name)
                refs.append(inp)
            counts["hessian_" + group] += 1
            with record_function("ablation/hessian_" + group):
                return original_h(inp, module, *args, **kw)

        def linear(*args, **kw):
            counts["linear"] += 1
            with record_function("ablation/model_linear"):
                return original_linear(*args, **kw)

        def wrap(hook):
            @wraps(hook)
            def imatrix(*args, **kw):
                counts["imatrix"] += 1
                with record_function("ablation/imatrix"):
                    return hook(*args, **kw)

            return imatrix

        with ExitStack() as stack:
            for m in names:
                hooks = m._forward_pre_hooks
                for k, hook in list(hooks.items()):
                    if (
                        getattr(hook, "__module__", "")
                        == "llmcompressor.observers.imatrix"
                    ):
                        hooks[k] = wrap(hook)
                        stack.callback(hooks.__setitem__, k, hook)
            stack.enter_context(patch.object(base, "accumulate_hessian", h))
            stack.enter_context(patch.object(torch.nn.functional, "linear", linear))
            with record_function("ablation/forward"):
                out = original_forward(graph, model, **kwargs)
        groups = list(shared.values())
        if refs:
            examples.append(refs[0])
        if not groups or not counts["imatrix"]:
            raise RuntimeError(
                f"Expected both expert Hessians and iMatrix hooks: {counts}"
            )
        records.append(
            {
                "counts": counts,
                "same_input_group_sizes": [len(g) for g in groups],
                "first_module": groups[0][0],
                "last_module": groups[-1][-1],
            }
        )
        return out

    with patch.object(pipeline, "_get_batches", get_batches), patch.object(
        Subgraph, "forward", forward
    ), patch.object(
        sys,
        "argv",
        [
            "run.py",
            "--mode",
            "auto",
            "--checkpoint",
            "granite",
            "--model-dir",
            str(a.model_dir),
            "--calibration-dir",
            str(a.calibration_dir),
            "--samples",
            "64",
            "--seq-length",
            "1024",
            "--coarse-timing",
            "--out",
            str(a.out / "oneshot.json"),
        ],
    ):
        run.main()
    if len(records) != 6:
        raise ValueError(f"Expected six sampled forwards, got {len(records)}")
    (a.out / "observations.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
