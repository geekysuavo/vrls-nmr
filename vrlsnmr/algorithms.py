import torch
from torch import Tensor

import vrlsnmr.operators as op


@torch.inference_mode()
def vrls(
    y: Tensor,
    ids: Tensor,
    tau: float,
    xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor]:
    """
    Basic VRLS.

    Args:
        y: :math:`(b, m)`, Measurements (complex).
        ids: :math:`(m)`, Measurement indices.
        tau: Fixed noise precision.
        xi: Fixed weight scale.
        n: Sparse dimension size.
        niter: Number of iterations.
        eps: Weight update positivity parameter.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Sparse mean (complex).
        - :math:`(b, n)`, Sparse variance (real, positive).
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    bs = y.size(dim=0)

    w = torch.ones((bs, n), dtype=real_dtype, device=device)
    mu = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    tau = torch.full((bs,), tau, dtype=real_dtype, device=device)

    for _ in range(niter):
        K = op.kernel(w, tau, ids)
        Kinv = torch.linalg.inv(K)

        mu.mul_(0)
        mu[:, ids] = (Kinv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = torch.fft.fft(mu, norm="ortho") / w

        Gamma_diag = op.xmarginal(Kinv, w, ids)

        m2 = mu.abs().square() + Gamma_diag
        w = (xi / (m2 + eps)).sqrt()

    return (mu, Gamma_diag)


@torch.inference_mode()
def vrls_ex(
    y: Tensor,
    ids: Tensor,
    beta_tau: float,
    beta_xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor]:
    """
    Extended VRLS.

    Args:
        y: :math:`(b, m)`, Measurements (complex).
        ids: :math:`(m)`, Measurement indices.
        beta_tau: Prior scale parameter for noise precision.
        beta_xi: Prior scale parameter for weight scale.
        n: Sparse dimension size.
        niter: Number of iterations.
        eps: Weight update positivity parameter.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Sparse mean (complex).
        - :math:`(b, n)`, Sparse variance (real, positive).
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    bs = y.size(dim=0)

    mu = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    nu_w = torch.ones((bs, n), dtype=real_dtype, device=device)
    nu_tau = torch.ones((bs,), dtype=real_dtype, device=device)
    nu_xi = torch.ones((bs,), dtype=real_dtype, device=device)

    for _ in range(niter):
        K = op.kernel(nu_w, nu_tau, ids)
        Kinv = torch.linalg.inv(K)

        mu.mul_(0)
        mu[:, ids] = (Kinv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = torch.fft.fft(mu, norm="ortho") / nu_w

        Gamma_diag = op.xmarginal(Kinv, nu_w, ids)
        Sigma_diag = op.ymarginal(Kinv, nu_w, ids)

        m2 = mu.abs().square() + Gamma_diag
        nu_w = (nu_xi.unsqueeze(dim=-1) / (m2 + eps)).sqrt()

        nu_xi = (beta_xi / nu_w.reciprocal().sum(dim=-1)).sqrt()

        yhat = torch.fft.ifft(mu, norm="ortho")[:, ids]
        err = (y - yhat).abs().square().sum(dim=-1)
        ess = err + Sigma_diag[:, ids].sum(dim=-1) / n
        nu_tau = (beta_tau / ess).sqrt()

    return (mu, Gamma_diag)


@torch.inference_mode()
def vrls_mf(
    y: Tensor,
    ids: Tensor,
    tau: float,
    xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor]:
    """
    Fast mean-field VRLS.

    Args:
        y: :math:`(b, m)`, Measurements (complex).
        ids: :math:`(m)`, Measurement indices.
        tau: Fixed noise precision.
        xi: Fixed weight scale.
        n: Sparse dimension size.
        niter: Number of iterations.
        eps: Weight update positivity parameter.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Sparse mean (complex).
        - :math:`(b, n)`, Sparse variance (real, positive).
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    bs = y.size(dim=0)
    m = ids.size(dim=0)

    w = torch.ones((bs, n), dtype=real_dtype, device=device)
    mu = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    gamma = torch.ones((bs, n), dtype=real_dtype, device=device)

    mask = torch.zeros((n,), dtype=torch.bool, device=device)
    mask[ids] = True

    AtA_diag = m / n
    L = 1

    for _ in range(niter):
        m2 = mu.abs().square() + gamma
        w = (xi / (m2 + eps)).sqrt()

        delta = torch.fft.ifft(mu, norm="ortho")
        delta[:, ~mask] = 0
        delta[:, mask] -= y
        delta = torch.fft.fft(delta, norm="ortho")
        delta = (L / 2) * mu - delta

        D_diag = tau / 2 + w
        mu = tau * D_diag.reciprocal() * delta

        gamma = (tau * AtA_diag + w).reciprocal()

    return (mu, gamma)
