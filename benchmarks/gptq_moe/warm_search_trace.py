"""Capture searches 29–56 of the real TinyMoE run after one whole decoder layer."""

from unittest.mock import patch

import torch

from llmcompressor.observers import grid_search, imatrix
from run import main

if __name__ == "__main__":
    original = imatrix._grid_search_observer
    qparams = grid_search.calculate_qparams
    count = 0

    def mark_qparams(*args, **kwargs):
        with torch.cuda.nvtx.range("GPTQ/base_qparams"):
            return qparams(*args, **kwargs)

    def search(*args, **kwargs):
        global count
        count += 1
        if not 29 <= count <= 56:
            return original(*args, **kwargs)
        if count == 29:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("GPTQ/scale_search"):
            result = original(*args, **kwargs)
        if count == 56:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        return result

    with patch.object(imatrix, "_grid_search_observer", search), patch.object(
        grid_search, "calculate_qparams", mark_qparams
    ):
        main()
    assert count == 280
