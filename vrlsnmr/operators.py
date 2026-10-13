"""Bindings for compiled operators."""

# Standard
from pathlib import Path

# Third party
import torch
from torch import Tensor

torch.ops.load_library(Path(__file__).parent / "_ops.so")

_default_dtype = torch.int64
_default_device = torch.get_default_device()


def kernel(weights: Tensor, tau: Tensor, ids: Tensor) -> Tensor:
    """
    Construct a kernel matrix.

    Args:
        weights: :math:`(b, n)`, Spectral weights.
        tau: :math:`(b)`, Noise precision.
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.

    Returns:
        :math:`(b, m, m)`, Kernel matrix.
    """
    if ids.ndim == 1:
        return torch.ops.vrlsnmr.kernel(weights, tau, ids)
    else:
        raise NotImplementedError


def xmarginal(kernel_inv: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
    """
    Compute frequency-domain marginal variances.

    Args:
        kernel_inv: :math:`(b, m, m)`, Inverse kernel matrix.
        weights: :math:`(b, n)`, Spectral weights.
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.

    Returns:
        :math:`(b, n)`, Frequency-domain marginal variances.
    """
    if ids.ndim == 1:
        return torch.ops.vrlsnmr.xmarginal(kernel_inv, weights, ids)
    else:
        raise NotImplementedError


def ymarginal(kernel_inv: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
    """
    Compute time-domain marginal variances.

    Args:
        kernel_inv: :math:`(b, m, m)`, Inverse kernel matrix.
        weights: :math:`(b, n)`, Spectral weights.
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.

    Returns:
        :math:`(b, n)`, Time-domain marginal variances.
    """
    if ids.ndim == 1:
        return torch.ops.vrlsnmr.ymarginal(kernel_inv, weights, ids)
    else:
        raise NotImplementedError


def schedunif(
    m: int,
    n: int | tuple[int, ...],
    dtype: torch.dtype = _default_dtype,
    device: torch.device = _default_device,
) -> Tensor:
    """
    Generate a uniform sampling schedule.

    Args:
        m: Number of grid points to sample.
        n: Total number of grid points, or grid shape.
        dtype: The desired data type of the returned tensor.
        device: The desired device of the returned tensor.

    Returns:
        :math:`(m)` or :math:`(m, d)`, Indices to sample.
    """
    if isinstance(n, int):
        perm = torch.randperm(n, dtype=dtype, device=device)
        return perm.narrow(dim=0, start=0, length=m)
    else:
        raise NotImplementedError


def schedexp(
    rate: float,
    m: int,
    n: int | tuple[int, ...],
    dtype: torch.dtype = _default_dtype,
    device: torch.device = _default_device,
) -> Tensor:
    """
    Generate an exponentially biased sampling schedule.

    Args:
        rate: Exponential biasing rate.
        m: Number of grid points to sample.
        n: Total number of grid points, or grid shape.
        dtype: The desired data type of the returned tensor.
        device: The desired device of the returned tensor.

    Returns:
        :math:`(m)` or :math:`(m, d)`, Indices to sample.
    """
    if isinstance(n, int):
        ids = torch.full((m,), fill_value=-1, dtype=dtype)
        torch.ops.vrlsnmr.schedexp(rate, n, ids)
        return ids.to(device)
    else:
        raise NotImplementedError


def schedpg(
    m: int,
    n: int | tuple[int, ...],
    dtype: torch.dtype = _default_dtype,
    device: torch.device = _default_device,
) -> Tensor:
    """
    Generate a Poisson-gap sampling schedule.

    Args:
        m: Number of grid points to sample.
        n: Total number of grid points, or grid shape.
        dtype: The desired data type of the returned tensor.
        device: The desired device of the returned tensor.

    Returns:
        :math:`(m)` or :math:`(m, d)`, Indices to sample.
    """
    if isinstance(n, int):
        ids = torch.full((m,), fill_value=-1, dtype=dtype)
        torch.ops.vrlsnmr.schedpg(n, ids)
        return ids.to(device)
    else:
        raise NotImplementedError
