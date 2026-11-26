"""One fixed-grid QuantEase coordinate sweep on a BF16 W4A16 projection."""

from typing import Literal

import torch
from compressed_tensors.offload import disable_offloading
from compressed_tensors.utils import update_offload_parameter

from llmcompressor.modifiers.gptq import GPTQModifier

__all__ = ["QuantEaseModifier", "coordinate_sweep"]


def coordinate_sweep(weight, initial, hessian, scales, group_size, tile=32):
    # Rows in q are input columns of the original Linear weight. Keeping output
    # channels contiguous makes each coordinate update a vector operation.
    q = initial.T.float().contiguous()
    w = weight.T.float().contiguous()
    s = scales.T.float().contiguous()
    width = q.size(0)
    if width % group_size or s.shape != (width // group_size, q.size(1)):
        raise ValueError("Expected a contiguous, grouped INT4 grid")
    diag = hessian.diagonal()
    for begin in range(0, width, tile):
        end = min(begin + tile, width)
        g = hessian[begin:end] @ (q - w)
        for j in range(begin, end):
            d = diag[j]
            target = q[j] - g[j - begin] / d.clamp_min(torch.finfo(d.dtype).tiny)
            scale = s[j // group_size]
            lower = torch.floor(target / scale).clamp(-8, 7)
            a = (lower * scale).to(weight.dtype).float()
            b = ((lower + 1).clamp_max(7) * scale).to(weight.dtype).float()
            best = torch.where((a - target).abs() <= (b - target).abs(), a, b)
            old = q[j]
            best = torch.where(
                (d > 0) & ((best - target).abs() < (old - target).abs()), best, old
            )
            if j + 1 < end:
                g[j - begin + 1 :].addr_(hessian[j, j + 1 : end], best - old)
            q[j].copy_(best)
    return q.T.to(weight.dtype).contiguous()


class QuantEaseModifier(GPTQModifier):
    """Apply one fixed-grid coordinate sweep after each BF16/grouped INT4 GPTQ fit.

    Modules are processed individually so that only one original weight and Hessian
    need to remain live during refinement. A module on GPTQ's RTN fallback keeps
    that initializer.
    """

    batched_quantization: Literal[1] | None = None

    def estimate_sequential_memory(self, module, num_tokens):
        held, scratch = super().estimate_sequential_memory(module, num_tokens)
        if not held:
            return held, scratch
        # Retained Gram, source BF16 weight, FP32 Q, W, Q-W and tile residual.
        size = module.weight.numel()
        rows = module.weight.shape[0]
        return held, scratch + 2 * held + 18 * size + 32 * rows * 4

    @torch.no_grad()
    def compress_module_list(self, module_list):
        for module in module_list:
            args = module.quantization_scheme.weights
            if (
                module.weight.dtype != torch.bfloat16
                or args.num_bits != 4
                or not args.symmetric
                or args.type != "int"
                or args.strategy != "group"
                or args.group_size <= 0
                or module.weight.shape[1] % args.group_size
            ):
                raise ValueError(
                    "QuantEase requires BF16 weights and symmetric grouped INT4"
                )
            with disable_offloading():
                source = module.weight.detach().clone()
                gram = self._hessians[module]
                count = self._num_samples[module].to(gram.device)
                if not torch.isfinite(count).all() or count.item() <= 0:
                    raise ValueError("QuantEase requires observed calibration samples")
                fallbacks_before = len(self._rtn_fallback_module_names)
                super().compress_module_list([module])
                if len(self._rtn_fallback_module_names) > fallbacks_before:
                    continue
                fitted = coordinate_sweep(
                    source,
                    module.weight.detach(),
                    gram.to(source.device) / count.to(source.device),
                    module.weight_scale,
                    args.group_size,
                )
                update_offload_parameter(module, "weight", fitted)
