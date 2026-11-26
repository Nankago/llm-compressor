"""Export a GPTQ fixture with an identity Hessian and fixed minmax scales."""

import sys
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.base import (
    PackedQuantizationCompressor,
)
from compressed_tensors.quantization import QuantizationScheme
from safetensors.torch import save_file

from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight


def main(out):
    # Two g128 groups per output row. Their extrema yield exactly these BF16 scales.
    values = torch.arange(128, device="cuda").float() / 16 - 4
    weight = torch.cat((values * 0.73, values), dim=0).to(torch.bfloat16)
    weight[0], weight[128] = 5.625, 7.5
    weight = weight.repeat(128, 1)
    scale = weight.unflatten(1, (2, 128)).abs().amax(2) / 7.5
    zero = torch.zeros_like(scale, dtype=torch.int8)
    scheme = QuantizationScheme.model_validate(
        {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "strategy": "group",
                "group_size": 128,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "strategy": "token",
                "dynamic": True,
            },
        }
    )
    fq, _, fallback = quantize_weight(
        weight[None],
        torch.eye(256, device="cuda")[None],
        scale[None],
        zero[None],
        None,
        scheme.weights,
    )
    assert not fallback.any()
    state = PackedQuantizationCompressor.compress(
        {"weight": fq[0], "weight_scale": scale, "weight_zero_point": zero}, scheme
    )
    save_file(
        {k: v.cpu().contiguous() for k, v in {**state, "source": weight}.items()}, out
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
