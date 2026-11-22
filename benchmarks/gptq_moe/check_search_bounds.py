"""Fingerprint CUDA grid choices on small packed, split and empty candidate grids."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.utils.impl_backend import ImplBackend

from llmcompressor.observers import grid_search  # noqa: F401 - register implementations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    result = {}
    cases = itertools.product([128, 1024], [False, True], [False, True], [0, 8])
    for group, symmetric, weighted, steps in cases:
        # Give each case random, zero and positive-constant groups.
        weights = torch.randn(1, 4, group, generator=torch.Generator().manual_seed(42))
        weights[:, 1] = 0
        weights[:, 2] = 1
        weights = weights.cuda().to(torch.bfloat16)
        importance = torch.linspace(0.1, 2.0, group, device="cuda").expand(4, -1)
        q = QuantizationArgs(
            num_bits=4, symmetric=symmetric, strategy="group", group_size=group
        )
        bounds = ImplBackend.call(
            "_grid_search_observer_triton",
            weights,
            q,
            0.8,
            0,
            steps / 0.8,
            2.4,
            importance_weights=importance if weighted else None,
        )
        sha = hashlib.sha256()
        for tensor in (*bounds, *calculate_qparams(*bounds, q)):
            sha.update(str(tensor.dtype).encode())
            sha.update(tensor.contiguous().view(torch.uint8).cpu().numpy())
        result[f"group{group}-symmetric{symmetric}-weighted{weighted}-steps{steps}"] = (
            sha.hexdigest()
        )
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
