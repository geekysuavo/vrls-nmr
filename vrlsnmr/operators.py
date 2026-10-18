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
    r"""
    Construct a kernel matrix.

    Args:
        weights: :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`, Spectral weights.
        tau: :math:`(b)`, Noise precision.
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.

    Returns:
        :math:`(b, m, m)` or :math:`(b, c, m, m)`, Kernel matrix.
    """
    if ids.ndim == 1:
        return torch.ops.vrlsnmr.kernel(weights, tau, ids)
    else:
        return torch.ops.vrlsnmr.kernel_nd(weights, tau, ids)


def xmarginal(kernel_inv: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
    r"""
    Compute frequency-domain marginal variances.

    Args:
        kernel_inv: :math:`(b, m, m)` or :math:`(b, c, m, m)`, Inverse kernel matrix.
        weights: :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`, Spectral weights.
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.

    Returns:
        :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
        Frequency-domain marginal variances.
    """
    if ids.ndim == 1:
        return torch.ops.vrlsnmr.xmarginal(kernel_inv, weights, ids)
    else:
        return torch.ops.vrlsnmr.xmarginal_nd(kernel_inv, weights, ids)


def ymarginal(kernel_inv: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
    r"""
    Compute time-domain marginal variances.

    Args:
        kernel_inv: :math:`(b, m, m)` or :math:`(b, c, m, m)`, Inverse kernel matrix.
        weights: :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`, Spectral weights.
        ids: :math:`(m)` or :math:`(m, d)`, Measurement indices.

    Returns:
        :math:`(b, n)` or :math:`(b, n_1, \dots, n_d)`,
        Time-domain marginal variances.
    """
    if ids.ndim == 1:
        return torch.ops.vrlsnmr.ymarginal(kernel_inv, weights, ids)
    else:
        return torch.ops.vrlsnmr.ymarginal_nd(kernel_inv, weights, ids)


def schedpg(
    m: int,
    n: int | tuple[int, ...],
    dtype: torch.dtype = _default_dtype,
    device: torch.device = _default_device,
) -> Tensor:
    """
    Generate a one-dimensional Poisson-gap sampling schedule.

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
        ids = torch.full((m, len(n)), fill_value=-1, dtype=dtype)
        torch.ops.vrlsnmr.schedpg_nd(n, ids)
        return ids.to(device)
