"""Opt-in single-process research patch for the measured all-expert forward.

Run this in place of run.py with identical arguments. No change to ordinary GPTQ.
"""

import json
import sys
from collections import Counter
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

import torch
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modifiers.gptq import base
from llmcompressor.modifiers.utils.hooks import HooksMixin

import run


def main():
    original_forward = LinearExperts2D.forward
    original_update = base.accumulate_hessian
    current = ContextVar("expert_gram_for_this_forward", default=None)
    counts = Counter()

    def forward(module, hidden_states, *args, **kwargs):
        if HooksMixin._HOOKS_DISABLED or not get_calibrate_all_experts_flag():
            return original_forward(module, hidden_states, *args, **kwargs)
        token = current.set({"input": hidden_states, "key": None})
        try:
            return original_forward(module, hidden_states, *args, **kwargs)
        finally:
            current.reset(token)

    def update(inp, module, hessian, num_samples, mask=None):
        slot = current.get()
        if (
            slot is None
            or slot["input"] is not inp
            or mask is not None
            or type(inp) is not torch.Tensor
            or inp.is_inference()
            or inp.ndim != 2
            or not isinstance(module, torch.nn.Linear)
            or inp.device != hessian.device
            or inp.device.type != "cuda"
            or hessian.dtype != torch.float32
        ):
            return original_update(inp, module, hessian, num_samples, mask=mask)
        # Identity is held strongly by this one invocation. In-place changes and
        # launches on a different stream invalidate the prior result.
        key = (inp._version, torch.cuda.current_stream(inp.device).cuda_stream)
        if slot["key"] != key:
            slot["gram"] = torch.zeros_like(hessian)
            slot["count"] = torch.zeros_like(num_samples)
            original_update(inp, module, slot["gram"], slot["count"])
            slot["key"] = key
            counts["computed"] += 1
        else:
            counts["reused"] += 1
        # Each module still owns its own accumulated Hessian and sample count.
        hessian.add_(slot["gram"])
        num_samples.add_(slot["count"])
        return hessian, num_samples

    with patch.object(LinearExperts2D, "forward", forward), patch.object(
        base, "accumulate_hessian", update
    ):
        run.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    if counts["computed"] == 0 or counts["reused"] == 0:
        raise ValueError(f"Prototype did not exercise reuse: {counts}")
    out.with_suffix(".reuse.json").write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    main()
