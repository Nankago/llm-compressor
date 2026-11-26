"""Compare the dense refit numerically to the pinned QDA implementation."""

import json
import subprocess
import sys
from pathlib import Path

import torch
from refit import refit


def main():
    qda = Path(sys.argv[1]).resolve()
    rev = subprocess.check_output(
        ["git", "-C", str(qda), "rev-parse", "HEAD"], text=True
    ).strip()
    if rev != "dc39a02bfd87bb2790e34001708d54ce9131e076":
        raise ValueError("Expected the pinned QDA revision")
    sys.path.insert(0, str(qda / "packages/qda-quant/src"))
    from qda_quant.steps.gptaq import refit as reference

    fixture = []
    for width, rows in ((8, 11), (32, 19), (128, 256)):
        rng = torch.Generator().manual_seed(12 + width)
        q = torch.randn(3 * width, width, generator=rng).bfloat16().float()
        teacher = (q + 0.18 * torch.randn(q.shape, generator=rng)).bfloat16().float()
        w = torch.randn(rows, width, generator=rng).bfloat16()
        h = q.T @ q / len(q)
        c = (teacher - q).T @ q / len(q)
        got = refit(w, h, c, 0.3, 0.01)
        expected = reference(w[None], h[None], c[None], 0.3, 0.01)[0].to(w.dtype)
        assert torch.equal(got, expected)
        fixture.append(dict(width=width, rows=rows, equal=True))
    print(json.dumps(dict(qda=rev, fixtures=fixture)))


if __name__ == "__main__":
    main()
