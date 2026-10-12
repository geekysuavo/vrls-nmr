"""Sampling and reconstruction algorithms."""

# Third party
import torch
from torch import Tensor

# VRLS
import vrlsnmr.operators as op
from vrlsnmr.types import (
    AlgorithmFunc,
    AlgorithmReturnType,
    CallbackFunc,
    MeasurementFunc,
    PointEstimatorReturnType,
    SamplerReturnType,
)


def ans(
    model: AlgorithmFunc,
    measure: MeasurementFunc,
    m_initial: int,
    m_final: int,
    n_initial: int,
    n_final: int,
    min_sparsity: float,
    warm_start: bool = False,
    callback: CallbackFunc | None = None,
    **kwargs: bool | float,
) -> SamplerReturnType:
    """
    Active nonuniform sampling (ANS).

    Args:
        model: Reconstruction function accepting measured data with indices
            and returning frequency- and time-domain means and variances.
            See :func:`vrls` for a candidate function.
        measure: Measurement function accepting a grid point to observe
            and returning a :math:`(1)` or :math:`(b, 1)` complex tensor.
        m_initial: Initial number of (uniform) measurements.
        m_final: Final (total) number of measurements.
        n_initial: Initial (minimum) Fourier grid size.
        n_final: Final (maximum) Fourier grid size.
        min_sparsity: Minimum sparsity maintained for :attr:`model`.
        warm_start: When true, model weights are carried between steps.
        callback: Optional function to call at the end of each step.
        kwargs: Keyword arguments passed to :attr:`model`.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, m)`, Time-domain measurements (complex).
        - :math:`(m)`, Time-domain measurement indices.
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

    w = True
    state = None
    while m < m_final:
        m = ids.numel()
        n = adapt_size(m, n)

        if warm_start:
            _out = model(y, ids, n=n, warm_start=w, **kwargs)
            (mu, gamma_diag, yhat, sigma_diag, w) = _out
        else:
            (mu, gamma_diag, yhat, sigma_diag) = model(y, ids, n=n, **kwargs)

        mu = mu.roll(n // 2, dims=1)
        gamma_diag = gamma_diag.roll(n // 2, dims=1)

        yhat = yhat.narrow(dim=1, start=0, length=n // 2)
        sigma_diag = sigma_diag.narrow(dim=1, start=0, length=n // 2)

        objective = sigma_diag.sum(dim=0).clone()
        objective[ids] = 0

        next_index = objective.argmax(keepdim=True)
        y_next = measure(int(next_index.item()))
        ids = torch.cat((ids, next_index))
        y = torch.cat((y, y_next), dim=-1)

        state = (y, ids, mu, gamma_diag, yhat, sigma_diag)
        if callback is not None:
            callback(state)

    assert state is not None
    return state


@torch.inference_mode()
def vrls(
    y: Tensor,
    ids: Tensor,
    tau: float,
    xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
    warm_start: bool | Tensor = False,
) -> AlgorithmReturnType:
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
        warm_start: Optional initial weights, or whether to return final weights.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Frequency-domain mean (complex).
        - :math:`(b, n)`, Frequency-domain variance (real, positive).
        - :math:`(b, n)`, Time-domain mean (complex).
        - :math:`(b, n)`, Time-domain variance (real, positive).
        - :math:`(b, n)`, Frequency-domain weights (real, positive), only returned
          when :attr:`warm_start` is true or a tensor.
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    bs = y.size(dim=0)

    mu = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    tau_batched = torch.full((bs,), tau, dtype=real_dtype, device=device)

    if torch.is_tensor(warm_start):
        w = warm_start
        return_weights = True
    else:
        w = torch.ones((bs, n), dtype=real_dtype, device=device)
        return_weights = warm_start

    for _ in range(niter):
        K = op.kernel(w, tau_batched, ids)
        Kinv = torch.linalg.inv(K)

        mu.mul_(0)
        mu[:, ids] = (Kinv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = torch.fft.fft(mu, norm="ortho") / w

        gamma_diag = op.xmarginal(Kinv, w, ids)

        m2 = mu.abs().square() + gamma_diag
        w = (xi / (m2 + eps)).sqrt()

    yhat = torch.fft.ifft(mu, norm="ortho")
    sigma_diag = op.ymarginal(Kinv, w, ids)

    if return_weights:
        return (mu, gamma_diag, yhat, sigma_diag, w)
    else:
        return (mu, gamma_diag, yhat, sigma_diag)


@torch.inference_mode()
def vrls_ex(
    y: Tensor,
    ids: Tensor,
    beta_tau: float,
    beta_xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
    warm_start: bool | Tensor = False,
) -> AlgorithmReturnType:
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
        warm_start: Optional initial weights, or whether to return final weights.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Frequency-domain mean (complex).
        - :math:`(b, n)`, Frequency-domain variance (real, positive).
        - :math:`(b, n)`, Time-domain mean (complex).
        - :math:`(b, n)`, Time-domain variance (real, positive).
        - :math:`(b, n)`, Frequency-domain weights (real, positive), only returned
          when :attr:`warm_start` is true or a tensor.
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    bs = y.size(dim=0)

    mu = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    nu_tau = torch.ones((bs,), dtype=real_dtype, device=device)
    nu_xi = torch.ones((bs,), dtype=real_dtype, device=device)

    if torch.is_tensor(warm_start):
        nu_w = warm_start
        return_weights = True
    else:
        nu_w = torch.ones((bs, n), dtype=real_dtype, device=device)
        return_weights = warm_start

    for _ in range(niter):
        K = op.kernel(nu_w, nu_tau, ids)
        Kinv = torch.linalg.inv(K)

        mu.mul_(0)
        mu[:, ids] = (Kinv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = torch.fft.fft(mu, norm="ortho") / nu_w

        gamma_diag = op.xmarginal(Kinv, nu_w, ids)
        sigma_diag = op.ymarginal(Kinv, nu_w, ids)

        m2 = mu.abs().square() + gamma_diag
        nu_w = (nu_xi.unsqueeze(dim=-1) / (m2 + eps)).sqrt()

        nu_xi = (beta_xi / nu_w.reciprocal().sum(dim=-1)).sqrt()

        yhat = torch.fft.ifft(mu, norm="ortho")
        err = (y - yhat[:, ids]).abs().square().sum(dim=-1)
        ess = err + sigma_diag[:, ids].sum(dim=-1) / n
        nu_tau = (beta_tau / ess).sqrt()

    if return_weights:
        return (mu, gamma_diag, yhat, sigma_diag, nu_w)
    else:
        return (mu, gamma_diag, yhat, sigma_diag)


@torch.inference_mode()
def vrls_mf(
    y: Tensor,
    ids: Tensor,
    tau: float,
    xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
    warm_start: bool | Tensor = False,
) -> AlgorithmReturnType:
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
        warm_start: Optional initial weights, or whether to return final weights.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Frequency-domain mean (complex).
        - :math:`(b, n)`, Frequency-domain variance (real, positive).
        - :math:`(b, n)`, Time-domain mean (complex).
        - :math:`(b, n)`, Time-domain variance (real, positive).
        - :math:`(b, n)`, Frequency-domain weights (real, positive), only returned
          when :attr:`warm_start` is true or a tensor.
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    (bs, m) = y.shape

    mu = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    gamma = torch.ones((bs, n), dtype=real_dtype, device=device)

    if torch.is_tensor(warm_start):
        w = warm_start
        return_weights = True
    else:
        w = torch.ones((bs, n), dtype=real_dtype, device=device)
        return_weights = warm_start

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

    if return_weights:
        return (mu, gamma, yhat, sigma, w)
    else:
        return (mu, gamma, yhat, sigma)


@torch.inference_mode()
def ists(
    y: Tensor,
    ids: Tensor,
    mu: float,
    n: int,
    niter: int,
) -> PointEstimatorReturnType:
    """
    Iterative Soft Thresholding (IST-S).

    Args:
        y: :math:`(m)` or :math:`(b, m)`, Measurements (complex).
        ids: :math:`(m)`, Measurement indices.
        mu: Thresholding factor.
        n: Sparse dimension size.
        niter: Number of iterations.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Frequency-domain point estimate (complex).
        - :math:`(b, n)`, Time-domain point estimate (complex).
    """
    device = y.device
    complex_dtype = y.dtype

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    (bs, _) = y.shape

    xhat = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    yhat = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    delta = torch.zeros((bs, n), dtype=complex_dtype, device=device)

    mask = torch.zeros((n,), dtype=torch.bool, device=device)
    mask[ids] = True

    for it in range(niter):
        delta[:, mask] = y - yhat[:, mask]
        xhat += torch.fft.fft(delta, norm="ortho")

        if it == 0:
            thresh = xhat.abs().max(dim=1, keepdim=True).values

        xhat_abs = xhat.abs()
        xhat = torch.where(
            xhat_abs <= thresh,
            0.0,
            xhat * (1 - thresh / xhat_abs),
        )

        yhat = torch.fft.ifft(xhat, norm="ortho")
        thresh *= mu

    return (xhat, yhat)
