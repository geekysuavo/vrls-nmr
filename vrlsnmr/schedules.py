"""Static sampling schedules."""

# Standard
import math

# Third party
import torch
from torch import Tensor

# VRLS
import vrlsnmr.multidim as md
import vrlsnmr.operators as op

_default_dtype = torch.int64
_default_device = torch.get_default_device()


def uniform(
    m: int,
    n: int | tuple[int, ...],
    *,
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
    if md.is_multidim(n):
        assert isinstance(n, tuple)
        perm = torch.randperm(math.prod(n), dtype=dtype, device=device)
        flat_ids = perm.narrow(dim=0, start=0, length=m)
        return md.unravel_ids(flat_ids, n)
    else:
        assert isinstance(n, int)
        perm = torch.randperm(n, dtype=dtype, device=device)
        return perm.narrow(dim=0, start=0, length=m)


def exponential(
    rate: float,
    m: int,
    n: int | tuple[int, ...],
    *,
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
    if md.is_multidim(n):
        assert isinstance(n, tuple)
        grid = (torch.arange(grid_size, dtype=dtype, device=device) for grid_size in n)
        grid = torch.cartesian_prod(*grid)
        weight = torch.exp(-rate * grid.float().norm(dim=1))
    else:
        assert isinstance(n, int)
        grid = torch.arange(n, dtype=dtype, device=device)
        weight = torch.exp(-rate * grid)

    index = torch.multinomial(weight, m, replacement=False).sort().values
    return grid[index]


def poisson_gap(
    m: int,
    n: int | tuple[int, ...],
    *,
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
    if md.is_multidim(n):
        assert isinstance(n, tuple)
        return op.schedpg_nd(m, n, dtype, device)
    else:
        assert isinstance(n, int)
        return op.schedpg(m, n, dtype, device)
