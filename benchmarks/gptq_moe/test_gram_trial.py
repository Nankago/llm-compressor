"""Exercise reuse and its fallbacks on one CUDA device without loading Granite."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import try_gram_reuse as trial


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class TestScopedGram(unittest.TestCase):
    @torch.no_grad()
    def test_repeated_mutated_masked_and_distinct_inputs(self):
        update = trial.base.accumulate_hessian
        layer = torch.nn.Linear(
            128, 128, bias=False, device="cuda", dtype=torch.bfloat16
        )
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        mask = torch.arange(32, device="cuda") % 2 == 0

        def exercise(_, shared):
            real = torch.zeros(128, 128, device="cuda")
            oracle = torch.zeros_like(real)
            count = torch.zeros((), device="cuda")
            expected_count = torch.zeros_like(count)

            def check(inp, mask=None):
                trial.base.accumulate_hessian(inp, layer, real, count, mask=mask)
                update(inp, layer, oracle, expected_count, mask=mask)
                self.assertTrue(
                    torch.equal(real.view(torch.uint8), oracle.view(torch.uint8))
                )
                self.assertTrue(torch.equal(count, expected_count))

            check(shared)
            check(shared)
            check(shared, mask)
            check(shared.clone())
            shared.add_(1)
            check(shared)
            check(shared)

        def fake_run():
            trial.LinearExperts2D.forward(None, x)
            # A later use of exactly the same object outside the expert call must miss.
            h = torch.zeros(128, 128, device="cuda")
            n = torch.zeros((), device="cuda")
            trial.base.accumulate_hessian(x, layer, h, n)
            self.assertEqual(n.item(), 1)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            with patch.object(
                trial, "get_calibrate_all_experts_flag", lambda: True
            ), patch.object(trial.LinearExperts2D, "forward", exercise), patch.object(
                trial.run, "main", fake_run
            ), patch.object(
                sys, "argv", ["run.py", "--out", str(out)]
            ):
                trial.main()
            self.assertEqual(
                json.loads(out.with_suffix(".reuse.json").read_text()),
                {"computed": 2, "reused": 2},
            )


if __name__ == "__main__":
    unittest.main()
