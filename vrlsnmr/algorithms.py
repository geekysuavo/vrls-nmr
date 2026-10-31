"""Sampling and reconstruction algorithms."""

# Standard
import math
from typing import overload

# Third party
import torch
from torch import Tensor

# VRLS
import vrlsnmr.multicomplex as mx
import vrlsnmr.multidim as md
import vrlsnmr.operators as op
from vrlsnmr.types import (
    AdaptSizeFunc,
    AdaptSizeIntFunc,
    AdaptSizeTupleFunc,
    AlgorithmFunc,
    AlgorithmReturnType,
    CallbackFunc,
    MeasurementFunc,
    PointEstimatorReturnType,
    SamplerReturnType,
)


@overload
def ans(
    model: AlgorithmFunc,
    measure: MeasurementFunc,
    adapt_size: AdaptSizeIntFunc,
    *,
    m_initial: int | Tensor,
    m_final: int,
    n_initial: int,
    n_final: int,
    warm_start: bool = False,
    callback: CallbackFunc | None = None,
    device: torch.device | None = None,
    **kwargs: bool | float,
) -> SamplerReturnType: ...


@overload
def ans(
    model: AlgorithmFunc,
    measure: MeasurementFunc,
    adapt_size: AdaptSizeTupleFunc,
    *,
    m_initial: int | Tensor,
    m_final: int,
    n_initial: tuple[int, ...],
    n_final: tuple[int, ...],
    warm_start: bool = False,
    callback: CallbackFunc | None = None,
    device: torch.device | None = None,
    **kwargs: bool | float,
) -> SamplerReturnType: ...


def ans(
    model: AlgorithmFunc,
    measure: MeasurementFunc,
    adapt_size: AdaptSizeFunc,
    *,
    m_initial: int | Tensor,
    m_final: int,
    n_initial: int | tuple[int, ...],
    n_final: int | tuple[int, ...],
    warm_start: bool = False,
    callback: CallbackFunc | None = None,
    device: torch.device | None = None,
    **kwargs: bool | float,
) -> SamplerReturnType:
    r"""
    Active nonuniform sampling (ANS).

    Args:
        model: Reconstruction function accepting measured data with indices
            and returning frequency- and time-domain means and variances.
            See :func:`vrls` for a candidate function.
        measure: Measurement function accepting a grid point to observe
            and returning a :math:`(1)` or :math:`(b, 1)` complex tensor.
            See :class:`vrlsnmr.simulators.Signal` for a candidate function.
        adapt_size: Grid size adaptation function accepting the current
            number of measurements and grid size, and returning a new grid size.
            See :class:`vrlsnmr.gridsizes.PowerOfTwo` for a candidate function.
        m_initial: Initial number of (uniform) measurements, or an initial
            tensor of measurement indices.
        m_final: Final (total) number of measurements.
        n_initial: Initial (minimum) Fourier grid size.
        n_final: Final (maximum) Fourier grid size.
        warm_start: When true, model weights are carried between steps.
        callback: Optional function to call at the end of each step.
        device: Device to place initial index tensors onto.
        kwargs: Keyword arguments passed to :attr:`model`.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, m)` or :math:`(b, c, m)`,
          Time-domain measurements (complex or multicomplex).
        - :math:`(m)` or :math:`(m, d)`,
          Time-domain measurement indices.
        - :math:`(b, n)` or :math:`(b, c, n_1, \dots, n_d)`,
          Frequency-domain mean (complex or multicomplex).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Frequency-domain variance (real, positive).
        - :math:`(b, n / 2)` or :math:`(b, c, n_1 / 2, \dots, n_d / 2)`,
          Time-domain mean (complex or multicomplex).
        - :math:`(b, n / 2)` or :math:`(b, n_1 / 2, \dots, n_d / 2)`,
          Time-domain variance (real, positive).
    """
    if torch.is_tensor(m_initial):
        ids = m_initial
        device = ids.device
    else:
        grid_ndim = len(n_initial) if isinstance(n_initial, tuple) else 1
        m_per_dim = max(1, round(m_initial ** (1 / grid_ndim)))
        ids_per_dim = [torch.arange(m_per_dim, device=device)] * grid_ndim
        ids = torch.cartesian_prod(*ids_per_dim)

    m_initial = ids.size(dim=0)
    (m, n) = (m_initial, n_initial)
    y = measure(ids)

    w = True
    state = None
    while m < m_final:
        m = ids.size(dim=0)
        n = adapt_size(m, n)  # ty: ignore[invalid-argument-type]

        if warm_start:
            _out = model(y, ids, n=n, warm_start=w, **kwargs)
            (mu, gamma_diag, yhat, sigma_diag, w) = _out
        else:
            (mu, gamma_diag, yhat, sigma_diag) = model(y, ids, n=n, **kwargs)

        mu = md.roll(mu, n)
        gamma_diag = md.roll(gamma_diag, n)

        n_td = md.narrow_shape(n)
        flat_ids = md.ravel_ids(ids, n_td)
        objective = sigma_diag.sum(dim=0).flatten().clone()
        objective[flat_ids] = 0
        del flat_ids

        next_flat_index = objective.argmax(keepdim=True)
        next_index = md.unravel_ids(next_flat_index, n_td)

        y_next = measure(next_index)
        y = torch.cat((y, y_next), dim=-1)
        ids = torch.cat((ids, next_index))

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
        - :math:`(b, n / 2)` or :math:`(b, c, n_1 / 2, \dots, n_d / 2)`,
          Time-domain mean (complex or multicomplex).
        - :math:`(b, n / 2)` or :math:`(b, n_1 / 2, \dots, n_d / 2)`,
          Time-domain variance (real, positive).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Frequency-domain weights (real, positive), only returned
          when :attr:`warm_start` is true or a tensor.
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    (y, flat_ids, bs, nch, grid, shape, multidim) = _prepare(y, ids, n)
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

        m2 = mx.abs(mu, square=True) / nch + gamma_diag
        w = (xi / (m2 + eps)).sqrt()

    yhat = md.narrow(mx.fftn(mu, inverse=True), n)
    sigma_diag = op.ymarginal(kernel_inv, w, ids, windowed=True)

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
    n: int | tuple[int, ...],
    niter: int,
    eps: float = 1.0e-9,
    warm_start: bool | Tensor = False,
) -> AlgorithmReturnType:
    r"""
    Extended VRLS.

    Args:
        y: :math:`(m)`, :math:`(c, m)`, :math:`(b, m)` or :math:`(b, c, m)`,
            Measurements (complex or multicomplex).
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.
        beta_tau: Prior scale parameter for noise precision.
        beta_xi: Prior scale parameter for weight scale.
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
        - :math:`(b, n / 2)` or :math:`(b, c, n_1 / 2, \dots, n_d / 2)`,
          Time-domain mean (complex or multicomplex).
        - :math:`(b, n / 2)` or :math:`(b, n_1 / 2, \dots, n_d / 2)`,
          Time-domain variance (real, positive).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Frequency-domain weights (real, positive), only returned
          when :attr:`warm_start` is true or a tensor.
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    (y, flat_ids, bs, nch, grid, shape, multidim) = _prepare(y, ids, n)
    var_scale = nch / math.prod(grid)
    as_nu_w = (bs,) + (1,) * len(grid)

    mu = torch.zeros(shape, dtype=complex_dtype, device=device)
    nu_tau = torch.ones((bs,), dtype=real_dtype, device=device)
    nu_xi = torch.ones((bs,), dtype=real_dtype, device=device)

    if torch.is_tensor(warm_start):
        nu_w = warm_start
        return_weights = True
    else:
        nu_w = torch.ones((bs, *grid), dtype=real_dtype, device=device)
        return_weights = warm_start

    for _ in range(niter):
        kernel = op.kernel(nu_w, nu_tau, ids)
        kernel_inv = torch.linalg.inv(kernel)

        mu = (kernel_inv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = md.scatter(mu, flat_ids, n)
        mu = mx.fftn(mu) / mx.expand(nu_w)

        gamma_diag = op.xmarginal(kernel_inv, nu_w, ids)
        sigma_diag = op.ymarginal(kernel_inv, nu_w, ids)

        m2 = mx.abs(mu, square=True) / nch + gamma_diag
        nu_w = (nu_xi.view(as_nu_w) / (m2 + eps)).sqrt()

        nu_xi = (beta_xi / nu_w.reciprocal().flatten(start_dim=1).sum(dim=-1)).sqrt()

        yhat = mx.fftn(mu, inverse=True)
        err = (y - md.gather(yhat, flat_ids, n)).abs().square()
        err = err.flatten(start_dim=1).sum(dim=-1)
        var = md.gather(sigma_diag, flat_ids, n).sum(dim=-1)
        ess = err + var * var_scale
        nu_tau = (beta_tau / ess).sqrt()

    yhat = md.narrow(yhat, n)
    sigma_diag = md.narrow(sigma_diag, n)

    if multidim:
        mu = mx.to_states(mu)
        yhat = mx.to_states(yhat)

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
        - :math:`(b, n / 2)` or :math:`(b, c, n_1 / 2, \dots, n_d / 2)`,
          Time-domain mean (complex or multicomplex).
        - :math:`(b, n / 2)` or :math:`(b, n_1 / 2, \dots, n_d / 2)`,
          Time-domain variance (real, positive).
        - :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
          Frequency-domain weights (real, positive), only returned
          when :attr:`warm_start` is true or a tensor.
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    (y, flat_ids, bs, nch, grid, shape, multidim) = _prepare(y, ids, n)
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
        m2 = mx.abs(mu, square=True) / nch + gamma
        w = (xi / (m2 + eps)).sqrt()

        delta = mx.fftn(mu, inverse=True)
        delta = md.gather(delta, flat_ids, n) - y
        delta = md.scatter(delta, flat_ids, n)
        delta = mx.fftn(delta)
        delta = (lipschitz / 2) * mu - delta

        btb_diag = mx.expand(tau / 2 + w)
        mu = tau * btb_diag.reciprocal() * delta

        gamma = (tau * ata_diag + w).reciprocal()

    grid_dims = tuple(range(1, gamma.ndim))
    yhat = md.narrow(mx.fftn(mu, inverse=True), n)
    sigma = md.narrow(gamma.mean(dim=grid_dims, keepdim=True).expand_as(gamma), n)

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

    (y, flat_ids, bs, _, grid, shape, multidim) = _prepare(y, ids, n)
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
) -> tuple[Tensor, Tensor, int, int, tuple[int, ...], tuple[int, ...], bool]:
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
        - Batch size.
        - Channel count.
        - Grid tensor subshape.
        - Full tensor shape.
        - True when multidimensional.
    """
    multidim = md.is_multidim(n)
    if (multidim and y.ndim == 2) or (not multidim and y.ndim == 1):
        y = y.unsqueeze(dim=0)

    bs = y.size(dim=0)
    flat_ids = md.ravel_ids(ids, n)
    grid = n if isinstance(n, tuple) else (n,)
    nch = mx.channels(len(grid))

    if multidim:
        shape = (bs, nch, *grid)
        y = mx.to_complex(y)
    else:
        shape = (bs, *grid)

    return (y, flat_ids, bs, nch, grid, shape, multidim)
