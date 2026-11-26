import pytest
import torch
from refit import refit


def test_refit_against_ridge_least_squares():
    gen = torch.Generator().manual_seed(51)
    x = torch.randn(40, 8, generator=gen)
    teacher = x + 0.21 * torch.randn(40, 8, generator=gen)
    w = torch.randn(11, 8, generator=gen)
    h = x.T @ x
    cross = (teacher - x).T @ x
    lam = 0.01 * h.diagonal().mean()
    a = torch.cat([x.double(), lam.sqrt() * torch.eye(8, dtype=torch.float64)])
    b = torch.cat([(teacher.double() @ w.double().T), lam.sqrt() * w.double().T])
    exact = torch.linalg.lstsq(a, b).solution.T
    out = refit(w, h, cross, 1)
    torch.testing.assert_close(out.double(), exact, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(refit(w, h, cross, 0.3), w + 0.3 * (out - w))
    assert torch.equal(refit(w, h, torch.zeros_like(cross)), w)
    assert refit(w, None, None, 0) is w
    assert refit(w.bfloat16(), h, cross).dtype == torch.bfloat16
    with pytest.raises(ValueError):
        refit(w, h, cross * float("nan"))
    with pytest.raises(torch.linalg.LinAlgError):
        refit(w, torch.zeros_like(h), cross)


def test_cross_orientation():
    x = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, -4.0]])
    teacher = x @ torch.tensor([[1.0, 0.5], [-0.25, 1.5]])
    w = torch.tensor([[2.0, 3.0]])
    fit = refit(w, x.T @ x, (teacher - x).T @ x, 1, 0)
    torch.testing.assert_close(x @ fit.T, teacher @ w.T)
