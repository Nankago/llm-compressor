"""CPU-only numerical and cache protocol checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate import (
    binned,
    collate,
    forward,
    identity,
    prepare,
    read_rows,
    score,
    teacher_top,
    validate,
)


class Lookup(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(logits=self.table[input_ids])


class EvaluationTests(unittest.TestCase):
    def test_identity_including_zero_tail(self):
        p = torch.tensor([[0.1, 0.9, 0.0]], dtype=torch.float64).log()
        for k in (1, 2, 3):
            self.assertEqual(binned(teacher_top(p, k), p).item(), 0.0)

    def test_lower_bound_and_full_vocab(self):
        rng = torch.Generator().manual_seed(24)
        p, q = [
            torch.randn(23, 47, generator=rng, dtype=torch.float64).softmax(-1)
            for _ in range(2)
        ]
        exact = torch.distributions.kl_divergence(
            torch.distributions.Categorical(probs=p),
            torch.distributions.Categorical(probs=q),
        )
        prev = torch.zeros_like(exact)
        for k in (1, 9, 46, 47):
            out = binned(teacher_top(p.log(), k), q.log())
            self.assertTrue(
                (out >= prev - 1e-12).all() and (out <= exact + 1e-12).all()
            )
            prev = out
        torch.testing.assert_close(prev, exact, rtol=0, atol=1e-12)

    def test_teacher_ids_and_tail(self):
        p, q = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float64)
        ref = teacher_top(p[None].log(), 1)
        self.assertEqual(ref["ids"].item(), 0)
        expected = 0.7 * torch.log(
            torch.tensor(7.0, dtype=torch.float64)
        ) + 0.3 * torch.log(torch.tensor(1 / 3, dtype=torch.float64))
        torch.testing.assert_close(binned(ref, q[None].log())[0], expected)
        lp = torch.tensor([[0.0, -100.0, -101.0]], dtype=torch.float64).log_softmax(-1)
        self.assertGreater(teacher_top(lp, 1)["tail_logmass"].exp().item(), 0)
        self.assertTrue(
            torch.isinf(
                binned(ref, torch.tensor([[-torch.inf, 0.0, -torch.inf]]))
            ).item()
        )

    def test_positive_underflow_against_impossible_token(self):
        from evaluate import terms

        p = torch.tensor([-900.0, -torch.inf], dtype=torch.float64)
        q = torch.full_like(p, -torch.inf)
        self.assertEqual(terms(p, q).tolist(), [float("inf"), 0.0])

    def test_padding_target_offset_and_cache(self):
        rows = [("long", [2, 3, 1, 4], [1, 3]), ("short", [4, 2], [1])]
        ids, mask, ri, ti = collate(rows, 0, "cpu")
        self.assertEqual(ids.tolist(), [[2, 3, 1, 4], [4, 2, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(list(zip(ri.tolist(), ti.tolist())), [(0, 0), (0, 2), (1, 0)])
        p = torch.eye(5, dtype=torch.float64)
        q = p.roll(1, dims=0) * 2
        torch.testing.assert_close(
            forward(Lookup(p), rows, 0), p[[2, 1, 4]].log_softmax(-1)
        )
        tok = {"length": 5, "pad_id": 0}
        ident = identity(rows, tok, "teacher", "rev", 5)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            manifest = prepare(Lookup(p), rows, ident, cache, 5, 1)
            actual = score(Lookup(q), rows, ident, cache)
            pl, ql = [t[[2, 1, 4]].log_softmax(-1) for t in (p, q)]
            self.assertAlmostEqual(
                actual, (pl.exp() * (pl - ql)).sum(-1).mean().item(), 12
            )
            for key in (
                "format_version",
                "vocab_size",
                "reference_model",
                "reference_revision",
                "tokenizer",
                "inputs_sha256",
                "positions_sha256",
            ):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    validate({**manifest, key: None}, ident)
            changed = [("long", rows[0][1], [2, 3]), rows[1]]
            with self.assertRaisesRegex(ValueError, "positions_sha256"):
                score(
                    Lookup(q),
                    changed,
                    identity(changed, tok, "teacher", "rev", 5),
                    cache,
                )
            (cache / "000000.safetensors").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "checksum"):
                score(Lookup(q), rows, ident, cache)

    def test_invalid_target_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows"
            for positions in ([], [0], [3], [2, 1], [1, 1]):
                path.write_text(
                    json.dumps(
                        {
                            "id": "x",
                            "input_ids": [1, 2, 0],
                            "score_positions": positions,
                        }
                    )
                )
                with self.subTest(pos=positions), self.assertRaises(ValueError):
                    read_rows(path)


if __name__ == "__main__":
    unittest.main()
