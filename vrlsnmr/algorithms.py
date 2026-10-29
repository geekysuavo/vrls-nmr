"""Sampling and reconstruction algorithms."""

# Standard
import math

# Third party
import torch
from torch import Tensor

# VRLS
import vrlsnmr.multicomplex as mx
import vrlsnmr.multidim as md
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
    n: int | tuple[int, ...],
    niter: int,
    eps: float = 1.0e-9,
    warm_start: bool | Tensor = False,
) -> AlgorithmReturnType:
    r"""
    Basic VRLS.

    Args:
        y: :math:(m)`, :math:`(c, m)`, :math:`(b, m)` or :math:`(b, c, m)`,
            Measurements (complex or multicomplex).
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.
        tau: Fixed noise precision.
        xi: Fixed weight scale.
        n: Sparse dimension size or shape.
        niter: Number of iterations.
        eps: Weight update positivity parameter.
        warm_start: Optional initial weights, or whether to return final weights.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)` or :math:`(b, c, n_1, \dots, n_d)`,
          Frequency-domain mean (complex or multicomplex).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Frequency-domain variance (real, positive).
        - :math:`(b, n)` or :math:`(b, c, n_1, \dots, n_d)`,
          Time-domain mean (complex or multicomplex).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Time-domain variance (real, positive).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Frequency-domain weights (real, positive), only returned
          when :attr:`warm_start` is true or a tensor.
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    (y, flat_ids, shape, grid, bs, multidim) = _prepare(y, ids, n)
    mu = torch.zeros(shape, dtype=complex_dtype, device=device)
    tau_batched = torch.full((bs,), tau, dtype=real_dtype, device=device)

    if torch.is_tensor(warm_start):
        w = warm_start
        return_weights = True
    else:
        w = torch.ones((bs, *grid), dtype=real_dtype, device=device)
        return_weights = warm_start

    for _ in range(niter):
        kernel = op.kernel(w, tau_batched, ids)
        kernel_inv = torch.linalg.inv(kernel)

        mu = (kernel_inv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = md.scatter(mu, flat_ids, n)
        mu = mx.fftn(mu) / mx.expand(w)

        gamma_diag = op.xmarginal(kernel_inv, w, ids)

        m2 = mx.abs(mu, square=True) + gamma_diag
        w = (xi / (m2 + eps)).sqrt()

    yhat = mx.fftn(mu, inverse=True)
    sigma_diag = op.ymarginal(kernel_inv, w, ids)

    if multidim:
        mu = mx.to_states(mu)
        yhat = mx.to_states(yhat)

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

    flat_ids = md.ravel_ids(ids, shape=n)
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
        kernel = op.kernel(nu_w, nu_tau, ids)
        kernel_inv = torch.linalg.inv(kernel)

        mu = (kernel_inv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = md.scatter(mu, flat_ids, n)
        mu = mx.fftn(mu) / mx.expand(nu_w)

        gamma_diag = op.xmarginal(kernel_inv, nu_w, ids)
        sigma_diag = op.ymarginal(kernel_inv, nu_w, ids)

        m2 = mu.abs().square() + gamma_diag
        nu_w = (nu_xi.unsqueeze(dim=-1) / (m2 + eps)).sqrt()

        nu_xi = (beta_xi / nu_w.reciprocal().sum(dim=-1)).sqrt()

        yhat = mx.fftn(mu, inverse=True)
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
    n: int | tuple[int, ...],
    niter: int,
    eps: float = 1.0e-9,
    warm_start: bool | Tensor = False,
) -> AlgorithmReturnType:
    r"""
    Fast mean-field VRLS.

    Args:
        y: :math:`(m)`, :math:`(c, m)`, :math:`(b, m)` or :math:`(b, c, m)`,
            Measurements (complex or multicomplex).
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.
        tau: Fixed noise precision.
        xi: Fixed weight scale.
        n: Sparse dimension size or shape.
        niter: Number of iterations.
        eps: Weight update positivity parameter.
        warm_start: Optional initial weights, or whether to return final weights.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)` or :math:`(b, c, n_1, \dots, n_d)`,
          Frequency-domain mean (complex or multicomplex).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Frequency-domain variance (real, positive).
        - :math:`(b, n)` or :math:`(b, c, n_1, \dots, n_d)`,
          Time-domain mean (complex or multicomplex).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Time-domain variance (real, positive).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Frequency-domain weights (real, positive), only returned
          when :attr:`warm_start` is true or a tensor.
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    (y, flat_ids, shape, grid, bs, multidim) = _prepare(y, ids, n)
    mu = torch.zeros(shape, dtype=complex_dtype, device=device)
    gamma = torch.ones((bs, *grid), dtype=real_dtype, device=device)

    if torch.is_tensor(warm_start):
        w = warm_start
        return_weights = True
    else:
        w = torch.ones((bs, *grid), dtype=real_dtype, device=device)
        return_weights = warm_start

    ata_diag = y.size(dim=-1) / math.prod(grid)  # m / n
    lipschitz = 1

    for _ in range(niter):
        m2 = mx.abs(mu, square=True) + gamma
        w = (xi / (m2 + eps)).sqrt()

        delta = mx.fftn(mu, inverse=True)
        delta = md.gather(delta, flat_ids, n) - y
        delta = md.scatter(delta, flat_ids, n)
        delta = mx.fftn(delta)
        delta = (lipschitz / 2) * mu - delta

        btb_diag = mx.expand(tau / 2 + w)
        mu = tau * btb_diag.reciprocal() * delta

        gamma = (tau * ata_diag + w).reciprocal()

    yhat = mx.fftn(mu, inverse=True)
    grid_dims = tuple(range(1, gamma.ndim))
    sigma = gamma.mean(dim=grid_dims, keepdim=True).expand_as(gamma)

    if multidim:
        mu = mx.to_states(mu)
        yhat = mx.to_states(yhat)

    if return_weights:
        return (mu, gamma, yhat, sigma, w)
    else:
        return (mu, gamma, yhat, sigma)


@torch.inference_mode()
def ists(
    y: Tensor,
    ids: Tensor,
    mu: float,
    n: int | tuple[int, ...],
    niter: int,
) -> PointEstimatorReturnType:
    r"""
    Iterative Soft Thresholding (IST-S).

    Args:
        y: :math:`(m)`, :math:`(c, m)`, :math:`(b, m)` or :math:`(b, c, m)`,
            Measurements (complex or multicomplex).
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.
        mu: Thresholding factor.
        n: Sparse dimension size or shape.
        niter: Number of iterations.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)` or :math:`(b, c, n_1, \dots, n_d)`,
          Frequency-domain point estimate (complex or multicomplex).
        - :math:`(b, n)` or :math:`(b, c, n_1, \dots, n_d)`,
          Time-domain point estimate (complex or multicomplex).
    """
    device = y.device
    complex_dtype = y.dtype

    (y, flat_ids, shape, grid, bs, multidim) = _prepare(y, ids, n)
    xhat = torch.zeros(shape, dtype=complex_dtype, device=device)
    yhat = torch.zeros(shape, dtype=complex_dtype, device=device)
    delta = torch.zeros(shape, dtype=complex_dtype, device=device)

    for it in range(niter):
        delta = y - md.gather(yhat, flat_ids, n)
        delta = md.scatter(delta, flat_ids, n)
        xhat += mx.fftn(delta)

        xhat_abs = mx.abs(xhat, keepdim=True)
        if it == 0:
            thresh = xhat_abs.flatten(start_dim=1).max(dim=1, keepdim=True).values
            if multidim:
                thresh = thresh.view(bs, 1, *([1] * len(grid)))

        xhat = torch.where(
            xhat_abs <= thresh,
            0.0,
            xhat * (1 - thresh / xhat_abs),
        )

        yhat = mx.fftn(xhat, inverse=True)
        thresh *= mu

    if multidim:
        xhat = mx.to_states(xhat)
        yhat = mx.to_states(yhat)

    return (xhat, yhat)


def _prepare(
    y: Tensor,
    ids: Tensor,
    n: int | tuple[int, ...],
) -> tuple[Tensor, Tensor, tuple[int, ...], tuple[int, ...], int, bool]:
    """
    Prepare data for reconstruction.

    Args:
        y: Optionally batched measurements.
        ids: Measurement indices.
        n: Grid size or shape.

    Returns:
        A :class:`tuple` containing

        - Batched measurements.
        - Flattened indices.
        - Full tensor shape.
        - Grid tensor subshape.
        - Batch size.
        - True when multidimensional.
    """
    multidim = md.is_multidim(n)
    if (multidim and y.ndim == 2) or (not multidim and y.ndim == 1):
        y = y.unsqueeze(dim=0)

    bs = y.size(dim=0)
    flat_ids = md.ravel_ids(ids, n)
    grid = n if isinstance(n, tuple) else (n,)
    if multidim:
        shape = (bs, mx.channels(len(grid)), *grid)
    else:
        shape = (bs, *grid)

    return (y, flat_ids, shape, grid, bs, multidim)
