"""CPU checks for the paired formats, output-head identity and disk protocol."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from evaluate import binned, teacher_top
from paired import Projection, full_kl, make_identity, prepare, score


class ToyBackbone(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table.to(torch.bfloat16))

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(last_hidden_state=self.table[input_ids])


class ToyLM(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self, table, head):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="granitemoe", logits_scaling=2.0, vocab_size=5, hidden_size=3
        )
        self.model = ToyBackbone(table)
        self.head = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
        self.head.weight.data.copy_(head)

    def get_output_embeddings(self):
        return self.head


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.path = Path(self.root.name)
        g = torch.Generator().manual_seed(41)
        table, head, other = [torch.randn(5, 3, generator=g) for _ in range(3)]
        self.teacher = ToyLM(table, head)
        self.candidate = ToyLM(other, head)
        self.rows = [("a", [1, 4, 3, 2], [1, 3]), ("b", [2, 1], [1])]
        self.tok = {"length": 5, "pad_id": 0}
        self.ident = make_identity(self.teacher, self.rows, self.tok, "toy", "r1")

    def test_formats_match_direct_at_selected_positions(self):
        direct = []
        for model in (self.teacher, self.candidate):
            # Independently select the logits preceding target tokens and apply Granite scaling.
            direct.append(
                (model.head(model.model.table[[1, 3, 2]]) / 2.0).double().softmax(-1)
            )
        expected = (
            torch.distributions.kl_divergence(
                torch.distributions.Categorical(probs=direct[0]),
                torch.distributions.Categorical(probs=direct[1]),
            )
            .mean()
            .item()
        )
        for kind, k in [("hidden", 5), ("topk", 5), ("topk", 2)]:
            with self.subTest(kind=kind, k=k):
                cache = self.path / f"{kind}-{k}"
                prepare(
                    self.teacher,
                    Projection(self.teacher),
                    self.rows,
                    self.ident,
                    cache,
                    kind,
                    k=k,
                    chunk=2,
                )
                val = score(
                    self.candidate,
                    Projection(self.candidate),
                    self.rows,
                    self.ident,
                    cache,
                    kind,
                )
                self.assertLessEqual(val, expected + 2e-7)
                if k == 5:
                    self.assertAlmostEqual(val, expected, places=6)
                self.assertEqual(
                    score(
                        self.teacher,
                        Projection(self.teacher),
                        self.rows,
                        self.ident,
                        cache,
                        kind,
                    ),
                    0.0,
                )

    def test_different_head_and_scale(self):
        cache = self.path / "A"
        teacher_head = Projection(self.teacher)
        prepare(self.teacher, teacher_head, self.rows, self.ident, cache, "hidden")
        with torch.no_grad():
            self.candidate.head.weight[0].add_(1.0)
        self.candidate.config.logits_scaling = 3.5
        other = Projection(self.candidate)
        with self.assertRaisesRegex(ValueError, "head mismatch"):
            score(self.candidate, other, self.rows, self.ident, cache, "hidden")
        value = score(
            self.candidate, other, self.rows, self.ident, cache, "hidden", teacher_head
        )
        p = teacher_head(self.teacher.model.table[[1, 3, 2]])
        q = other(self.candidate.model.table[[1, 3, 2]])
        self.assertAlmostEqual(value, full_kl(p, q).mean().item(), places=6)

    def test_cache_mismatches_and_storage(self):
        cache = self.path / "ref"
        proj = Projection(self.teacher)
        prepare(self.teacher, proj, self.rows, self.ident, cache, "hidden")
        for field in (
            "inputs_sha256",
            "positions_sha256",
            "reference_revision",
            "tokenizer",
            "format_version",
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                score(
                    self.teacher,
                    proj,
                    self.rows,
                    {**self.ident, field: None},
                    cache,
                    "hidden",
                )
        with self.assertRaisesRegex(ValueError, "kind"):
            score(self.teacher, proj, self.rows, self.ident, cache, "topk")
        shard = cache / "000000.safetensors"
        original = load_file(shard)
        save_file({"hidden": original["hidden"].float()}, shard)
        with self.assertRaisesRegex(ValueError, "checksum"):
            score(self.teacher, proj, self.rows, self.ident, cache, "hidden")
        manifest = cache / "manifest.json"
        obj = json.loads(manifest.read_text())
        obj["shards"][0] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "Invalid cache tensors"):
            score(self.teacher, proj, self.rows, self.ident, cache, "hidden")

    def test_zero_mass(self):
        p = torch.tensor([[0.0, -100.0, -torch.inf]]).log_softmax(-1)
        q = torch.tensor([[0.0, -torch.inf, -torch.inf]])
        self.assertEqual(full_kl(p, p).item(), 0.0)
        self.assertTrue(torch.isinf(full_kl(p, q)).item())
        self.assertEqual(binned(teacher_top(p, 3), p).item(), 0.0)
        # Positive log-mass underflows as a float32 probability against a zero Q.
        self.assertTrue(
            torch.isinf(full_kl(torch.tensor([[0.0, -1000.0]]), q[:, :2])).item()
        )


if __name__ == "__main__":
    unittest.main()
