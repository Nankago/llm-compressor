"""Add three host ranges to the existing benchmark in the trace process only."""

from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

import torch

from llmcompressor.modifiers.gptq import base
from llmcompressor.observers import imatrix


@contextmanager
def observer_ranges():
    def mark(fn, label):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with torch.cuda.nvtx.range(label):
                return fn(*args, **kwargs)

        return wrapped

    with ExitStack() as stack:
        for obj, name, label in (
            (base, "observe", "GPTQ/weight_observe"),
            (
                imatrix.IMatrixMSEObserver,
                "_get_validated_importance",
                "GPTQ/_get_validated_importance",
            ),
            # Patch the imported dispatch entry that this observer actually calls.
            # Its correlated kernels distinguish the packed, split and eager paths.
            (imatrix, "_grid_search_observer", "GPTQ/scale_search"),
        ):
            stack.enter_context(
                patch.object(obj, name, mark(getattr(obj, name), label))
            )
        yield


if __name__ == "__main__":
    from run import main

    with observer_ranges():
        main()
