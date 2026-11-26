import torch

from llmcompressor.modifiers.experimental.quantease import coordinate_sweep


def test_matches_exhaustive_coordinate_search():
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(54, 32, generator=gen)
    x[:, 9] = 0
    h = x.T @ x
    w = torch.randn(19, 32, generator=gen).bfloat16()
    scales = (torch.rand(19, 4, generator=gen) + 0.15).bfloat16()
    s = scales.repeat_interleave(8, dim=1)
    initial = ((w.float() / s.float()).round().clamp(-8, 7) * s.float()).bfloat16()
    got = coordinate_sweep(w, initial, h, scales, 8, tile=7)
    ref = initial.double().clone()
    for j in range(w.size(1)):
        levels = (s[:, j, None].float() * torch.arange(-8, 8)).bfloat16().double()
        diff = levels - ref[:, j, None]
        grad = (ref - w.double()) @ h[:, j].double()
        costs = 2 * diff * grad[:, None] + diff.square() * h[j, j].double()
        costs, idx = costs.min(-1)
        ref[:, j] = torch.where(
            costs < 0, levels.gather(1, idx[:, None]).squeeze(1), ref[:, j]
        )
    assert torch.equal(got.double(), ref)
    assert torch.equal(got[:, 9], initial[:, 9])
    assert not torch.equal(got, initial)


def test_rtn_fallback_is_left_to_gptq(monkeypatch):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    from llmcompressor.modifiers.experimental.quantease import QuantEaseModifier
    from llmcompressor.modifiers.gptq import GPTQModifier

    module = torch.nn.Linear(32, 8, bias=False, dtype=torch.bfloat16)
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=32,
        ),
    )
    mod = QuantEaseModifier(scheme="W4A16", targets=["Linear"])
    mod._hessians[module] = torch.eye(32)
    mod._num_samples[module] = torch.tensor(2)

    def fallback(self, modules):
        assert modules == [module]
        module.weight.fill_(1)
        self._rtn_fallback_module_names.append("projection")

    monkeypatch.setattr(GPTQModifier, "compress_module_list", fallback)
    mod.compress_module_list([module])
    assert torch.equal(module.weight, torch.ones_like(module.weight))
