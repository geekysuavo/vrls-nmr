from pathlib import Path

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
        ids: :math:`(m)`, Measurement indices.

    Returns:
        :math:`(b, m, m)`, Kernel matrix.
    """
    return torch.ops.vrlsnmr.kernel(weights, tau, ids)


def xmarginal(kernel_inv: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
    """
    Compute frequency-domain marginal variances.

    Args:
        kernel_inv: :math:`(b, m, m)`, Inverse kernel matrix.
        weights: :math:`(b, n)`, Spectral weights.
        ids: :math:`(m)`, Measurement indices.

    Returns:
        :math:`(b, n)`, Frequency-domain marginal variances.
    """
    return torch.ops.vrlsnmr.xmarginal(kernel_inv, weights, ids)


def ymarginal(kernel_inv: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
    """
    Compute time-domain marginal variances.

    Args:
        kernel_inv: :math:`(b, m, m)`, Inverse kernel matrix.
        weights: :math:`(b, n)`, Spectral weights.
        ids: :math:`(m)`, Measurement indices.

    Returns:
        :math:`(b, n)`, Time-domain marginal variances.
    """
    return torch.ops.vrlsnmr.ymarginal(kernel_inv, weights, ids)


def schedunif(
    m: int,
    n: int,
    dtype: torch.dtype = _default_dtype,
    device: torch.device = _default_device,
) -> Tensor:
    """
    Generate a uniform sampling schedule.

    Args:
        m: Number of grid points to sample.
        n: Total number of available grid points.
        dtype: The desired data type of the returned tensor.
        device: The desired device of the returned tensor.

    Returns:
        :math:`(m)`, Indices to sample.
    """
    return (
        torch.randperm(n, dtype=dtype, device=device)
        .narrow(dim=0, start=0, length=m)
    )


def schedexp(
    rate: float,
    m: int,
    n: int,
    dtype: torch.dtype = _default_dtype,
    device: torch.device = _default_device,
) -> Tensor:
    """
    Generate an exponentially biased sampling schedule.

    Args:
        rate: Exponential biasing rate.
        m: Number of grid points to sample.
        n: Total number of available grid points.
        dtype: The desired data type of the returned tensor.
        device: The desired device of the returned tensor.

    Returns:
        :math:`(m)`, Indices to sample.
    """
    ids = torch.full((m,), fill_value=-1, dtype=dtype)
    torch.ops.vrlsnmr.schedexp(rate, n, ids)
    return ids.to(device)


def schedpg(
    m: int,
    n: int,
    dtype: torch.dtype = _default_dtype,
    device: torch.device = _default_device,
) -> Tensor:
    """
    Generate a Poisson-gap sampling schedule.

    Args:
        m: Number of grid points to sample.
        n: Total number of available grid points.
        dtype: The desired data type of the returned tensor.
        device: The desired device of the returned tensor.

    Returns:
        :math:`(m)`, Indices to sample.
    """
    ids = torch.full((m,), fill_value=-1, dtype=dtype)
    torch.ops.vrlsnmr.schedpg(n, ids)
    return ids.to(device)
