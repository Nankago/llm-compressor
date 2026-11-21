"""In-process probes for the pinned upstream implementation (single CUDA device)."""

from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

import torch

from llmcompressor.modifiers.gptq import base, gptq_quantize, helpers
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential.helpers import Subgraph


class Trace:
    def __init__(self):
        self.stage = None
        self.start = None
        self.spans = []
        self.batches = []
        self.caps = []
        self.group = -1
        self.names = {}
        self.fallback = []
        self.calls = Counter()

    def switch(self, stage):
        """One boundary event ends the old interval and starts the new interval."""
        previous = self.stage
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        if previous is not None:
            self.spans.append((previous, self.start, event))
        self.start, self.stage = event, stage
        return previous

    def timed(self, fn, stage):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            previous = self.switch(stage)
            self.calls[stage] += 1
            try:
                return fn(*args, **kwargs)
            finally:
                self.switch(previous)

        return wrapper

    def seconds(self):
        assert self.stage is None
        totals = Counter()
        for stage, start, end in self.spans:
            totals[stage] += start.elapsed_time(end) / 1000
        return dict(totals)

    @contextmanager
    def install(self, observer_details=False):
        original_list = base.GPTQModifier.compress_module_list
        original_cap = helpers.max_batch_size
        original_prepare = base.prepare_batch
        original_forward = Subgraph.forward
        original_factor = gptq_quantize.factorize_hessian
        original_cholesky = torch.linalg.cholesky_ex
        original_fused = gptq_quantize.fused_gptq_block_update

        def compress_list(modifier, modules):
            self.group += 1
            self.names = modifier._module_names
            result = original_list(modifier, modules)
            self.fallback = list(modifier._rtn_fallback_module_names)
            return result

        def cap(module, mode, block_size):
            result = original_cap(module, mode, block_size)
            self.caps.append([self.group, self.names[module], result])
            return result

        def prepare(batch, qparams, hessians, counts):
            names = [self.names[m] for m in batch]
            shape = list(batch[0].weight.shape)
            h_shape = list(hessians[batch[0]].shape)
            assert all(list(m.weight.shape) == shape for m in batch)
            assert all(list(hessians[m].shape) == h_shape for m in batch)
            self.batches.append(
                {
                    "subgraph": self.group,
                    "size": len(batch),
                    "modules": names,
                    "projections": sorted({name.rsplit(".", 1)[1] for name in names}),
                    "weight_shape_each": shape,
                    "H_shape_each": h_shape,
                    "cholesky_ex_calls": 0,
                }
            )
            result = self.timed(original_prepare, "prepare_batch")(
                batch, qparams, hessians, counts
            )
            self.batches[-1]["H_shape_stacked"] = list(result[1].shape)
            return result

        def forward(graph, *args, **kwargs):
            stage = (
                "propagate_forward"
                if HooksMixin._HOOKS_DISABLED
                else "calib_forward_outside_gptq_hook"
            )
            return self.timed(original_forward, stage)(graph, *args, **kwargs)

        def factor(*args, **kwargs):
            assert self.stage == "quant_setup"
            self.switch("factor")
            result = original_factor(*args, **kwargs)
            self.switch("weight_update")
            return result

        def cholesky(H, *args, **kwargs):
            if self.stage == "factor":
                row = self.batches[-1]
                branch = {2: "per_item", 3: "batched"}[H.ndim]
                assert row.setdefault("cholesky", branch) == branch
                row["cholesky_ex_calls"] += 1
            return original_cholesky(H, *args, **kwargs)

        def fused(*args, **kwargs):
            self.calls["fused_block_update"] += 1
            return original_fused(*args, **kwargs)

        probes = [
            (base.GPTQModifier, "compress_module_list", compress_list),
            (helpers, "max_batch_size", cap),
            (base, "prepare_batch", prepare),
            (base, "quantize_weight", self.timed(base.quantize_weight, "quant_setup")),
            (gptq_quantize, "factorize_hessian", factor),
            (torch.linalg, "cholesky_ex", cholesky),
            (gptq_quantize, "fused_gptq_block_update", fused),
            (
                base,
                "update_batch_qparams",
                self.timed(base.update_batch_qparams, "writeback"),
            ),
            (Subgraph, "forward", forward),
            (
                base.GPTQModifier,
                "calibrate_module",
                self.timed(base.GPTQModifier.calibrate_module, "hook_setup"),
            ),
            (
                base,
                "accumulate_hessian",
                self.timed(base.accumulate_hessian, "accumulate"),
            ),
        ]
        if observer_details:
            # The imatrix collector is a closure registered in attach(). Identify
            # that upstream closure and preserve its normal disabled-hook behavior.
            original_register = torch.nn.Module.register_forward_pre_hook

            def register(module, hook, *args, **kwargs):
                if getattr(hook, "__module__", "") == "llmcompressor.observers.imatrix":
                    collector = hook
                    timed_collector = self.timed(collector, "imatrix_input_hook")

                    @wraps(collector)
                    def hook(*inputs, **options):
                        if HooksMixin._HOOKS_DISABLED:
                            return collector(*inputs, **options)
                        return timed_collector(*inputs, **options)

                return original_register(module, hook, *args, **kwargs)

            probes.extend(
                [
                    (torch.nn.Module, "register_forward_pre_hook", register),
                    (base, "observe", self.timed(base.observe, "weight_observe")),
                ]
            )
        with ExitStack() as stack:
            for obj, name, wrapper in probes:
                stack.enter_context(patch.object(obj, name, wrapper))
            yield
