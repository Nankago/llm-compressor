"""Replay identical logits-independent Hessian updates using captured expert inputs."""

import argparse
import json
from pathlib import Path

import torch
from llmcompressor.modifiers.gptq.helpers import accumulate_hessian
from safetensors.torch import load_file


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    groups = json.loads((a.trace / "observations.json").read_text())
    if any(row["same_input_group_sizes"] != [64] for row in groups):
        raise ValueError(
            "Replay requires one verified shared activation for 64 gate/up projections"
        )
    x = [
        v
        for i in (8, 40)
        for v in load_file(
            a.trace / f"replay-{i}.safetensors", device="cuda:0"
        ).values()
    ]
    features = x[0].shape[-1]
    layer = torch.nn.Linear(
        features, 512, bias=False, device="cuda", dtype=torch.bfloat16
    )

    def measure(reuse):
        # Each simulated projection retains an independent accumulator and counter.
        hs = torch.zeros(64, features, features, device="cuda").unbind()
        ns = torch.zeros(64, device="cuda").unbind()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for inp in x:
            if reuse:
                delta = torch.zeros_like(hs[0])
                count = torch.zeros_like(ns[0])
                accumulate_hessian(inp, layer, delta, count)
                for h, n in zip(hs, ns, strict=True):
                    h.add_(delta)
                    n.add_(count)
            else:
                for h, n in zip(hs, ns, strict=True):
                    accumulate_hessian(inp, layer, h, n)
        end.record()
        end.synchronize()
        return start.elapsed_time(end), hs, ns

    for reuse in (False, True):
        measure(reuse)
    times = []
    for rep in range(4):
        results = {}
        for reuse in ((False, True) if rep % 2 == 0 else (True, False)):
            results[reuse] = measure(reuse)
        native, shared = results[False], results[True]
        # Compare the underlying bits, including the sign of zero.
        for original, trial in zip(
            native[1] + native[2], shared[1] + shared[2], strict=True
        ):
            if not torch.equal(
                original.reshape(-1).view(torch.uint8),
                trial.reshape(-1).view(torch.uint8),
            ):
                raise ValueError("H or count differs")
        times.append({"native_ms": native[0], "reuse_ms": shared[0]})
    a.output.write_text(
        json.dumps(
            {
                "samples": len(x),
                "input_shape": list(x[0].shape),
                "fanout": 64,
                "all_H_and_counts_bitwise_equal": True,
                "pairs": times,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
