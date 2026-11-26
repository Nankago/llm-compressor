"""QDA-style damped cross-Hessian correction for a dense linear weight."""

import torch


def refit(weight, hessian, cross, alpha=0.3, damping=0.01):
    """Fit W to predict W times teacher inputs from aligned student inputs."""
    if not 0 <= alpha <= 1 or damping < 0:
        raise ValueError("Invalid interpolation factor or damping")
    if alpha == 0:
        return weight
    if hessian.shape != cross.shape or hessian.shape != (
        weight.size(1),
        weight.size(1),
    ):
        raise ValueError("Incompatible input-channel statistics")
    if not torch.isfinite(hessian).all() or not torch.isfinite(cross).all():
        raise ValueError("Nonfinite calibration statistics")
    gram = hessian.to(dtype=torch.float32, copy=True)
    gram.diagonal().add_(damping * gram.diagonal().mean())
    factor = torch.linalg.cholesky(gram)
    rhs = weight.float() @ cross.float()
    delta = torch.cholesky_solve(rhs.T.contiguous(), factor).T
    return (weight.float() + alpha * delta).to(weight.dtype)
