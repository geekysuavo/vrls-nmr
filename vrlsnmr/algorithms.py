from typing import Callable

import torch
from torch import Tensor

import vrlsnmr.operators as op


def ans(
    measure: Callable[[int], Tensor],
    m_initial: int,
    m_final: int,
    n_initial: int,
    n_final: int,
    min_sparsity: float,
    **vrls_kwargs: int | float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Active nonuniform sampling (ANS) using :func:`vrls`.

    Args:
        measure: Measurement function accepting a grid point to observe
            and returning a :math:`(1)` or :math:`(b, 1)` complex tensor.
        m_initial: Initial number of (uniform) measurements.
        m_final: Final (total) number of measurements.
        n_initial: Initial (minimum) Fourier grid size.
        n_final: Final (maximum) Fourier grid size.
        min_sparsity: Minimum sparsity maintained for :func:`vrls`.
        vrls_kwargs: Keyword arguments passed to :func:`vrls`.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, m)`, Final time-domain measurements (complex).
        - :math:`(m)`, Final time-domain measurement indices.
        - :math:`(b, n)`, Frequency-domain mean (complex).
        - :math:`(b, n)`, Frequency-domain variance (real, positive).
        - :math:`(b, n)`, Time-domain mean (complex).
        - :math:`(b, n)`, Time-domain variance (real, positive).
    """
    def adapt_size(m_next: int, n_prev: int) -> int:
        n_next = n_prev
        while n_next < n_final and m_next / n_next >= min_sparsity:
            n_next *= 2
        return n_next

    y = torch.cat([measure(t) for t in range(m_initial)], dim=-1)
    ids = torch.arange(m_initial, device=y.device)
    (m, n) = (m_initial, n_initial)

    while m < m_final:
        m = ids.numel()
        n = adapt_size(m, n)
        (mu, Gamma_diag, yhat, Sigma_diag) = vrls(y, ids, n=n, **vrls_kwargs)

        mu = mu.roll(n // 2, dims=1)
        Gamma_diag = Gamma_diag.roll(n // 2, dims=1)

        yhat = yhat.narrow(dim=1, start=0, length=n // 2)
        Sigma_diag = Sigma_diag.narrow(dim=1, start=0, length=n // 2)

        objective = Sigma_diag.sum(dim=0).clone()
        objective[ids] = 0

        next_index = objective.argmax(keepdim=True)
        y_next = measure(next_index.item())
        ids = torch.cat((ids, next_index))
        y = torch.cat((y, y_next), dim=-1)

    return (y, ids, mu, Gamma_diag, yhat, Sigma_diag)


@torch.inference_mode()
def vrls(
    y: Tensor,
    ids: Tensor,
    tau: float,
    xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Basic VRLS.

    Args:
        y: :math:(m)` or :math:`(b, m)`, Measurements (complex).
        ids: :math:`(m)`, Measurement indices.
        tau: Fixed noise precision.
        xi: Fixed weight scale.
        n: Sparse dimension size.
        niter: Number of iterations.
        eps: Weight update positivity parameter.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Frequency-domain mean (complex).
        - :math:`(b, n)`, Frequency-domain variance (real, positive).
        - :math:`(b, n)`, Time-domain mean (complex).
        - :math:`(b, n)`, Time-domain variance (real, positive).
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

    yhat = torch.fft.ifft(mu, norm="ortho")
    Sigma_diag = op.ymarginal(Kinv, w, ids)

    return (mu, Gamma_diag, yhat, Sigma_diag)


@torch.inference_mode()
def vrls_ex(
    y: Tensor,
    ids: Tensor,
    beta_tau: float,
    beta_xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Extended VRLS.

    Args:
        y: :math:`(m)` or :math:`(b, m)`, Measurements (complex).
        ids: :math:`(m)`, Measurement indices.
        beta_tau: Prior scale parameter for noise precision.
        beta_xi: Prior scale parameter for weight scale.
        n: Sparse dimension size.
        niter: Number of iterations.
        eps: Weight update positivity parameter.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Frequency-domain mean (complex).
        - :math:`(b, n)`, Frequency-domain variance (real, positive).
        - :math:`(b, n)`, Time-domain mean (complex).
        - :math:`(b, n)`, Time-domain variance (real, positive).
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

        yhat = torch.fft.ifft(mu, norm="ortho")
        err = (y - yhat[:, ids]).abs().square().sum(dim=-1)
        ess = err + Sigma_diag[:, ids].sum(dim=-1) / n
        nu_tau = (beta_tau / ess).sqrt()

    return (mu, Gamma_diag, yhat, Sigma_diag)


@torch.inference_mode()
def vrls_mf(
    y: Tensor,
    ids: Tensor,
    tau: float,
    xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Fast mean-field VRLS.

    Args:
        y: :math:`(m)` or :math:`(b, m)`, Measurements (complex).
        ids: :math:`(m)`, Measurement indices.
        tau: Fixed noise precision.
        xi: Fixed weight scale.
        n: Sparse dimension size.
        niter: Number of iterations.
        eps: Weight update positivity parameter.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Frequency-domain mean (complex).
        - :math:`(b, n)`, Frequency-domain variance (real, positive).
        - :math:`(b, n)`, Time-domain mean (complex).
        - :math:`(b, n)`, Time-domain variance (real, positive).
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    (bs, m) = y.shape

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

    yhat = torch.fft.ifft(mu, norm="ortho")
    sigma = gamma.mean(dim=1, keepdim=True).expand_as(yhat)

    return (mu, gamma, yhat, sigma)
