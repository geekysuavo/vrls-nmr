"""Utility functions for multidimensional NUS NMR grids."""

# Standard
import math

# Third party
import torch
from torch import Tensor


def is_multidim(shape: int | tuple[int, ...]) -> bool:
    """
    Return whether a shape is multidimensional.

    Args:
        shape: Number of grid points, or grid shape.

    Returns:
        Whether :attr:`shape` is multidimensional.
    """
    return isinstance(shape, tuple) and len(shape) > 1


def ravel_ids(ids: Tensor, shape: int | tuple[int, ...]) -> Tensor:
    """
    Convert multidimensional grid indices into flattened indices.

    Args:
        ids: :math:`(m, d)`, Multidimensional indices.
        shape: :math:`d`, Grid sizes along each dimension.

    Returns:
        :math:`(m)`, Flattened indices.
    """
    if is_multidim(shape):
        assert isinstance(shape, tuple)
    else:
        return ids

    flat_ids = ids.new_zeros(ids.size(dim=0))
    for dim, grid_size in enumerate(shape):
        flat_ids.mul_(grid_size)
        flat_ids.add_(ids[:, dim])

    return flat_ids


def unravel_ids(flat_ids: Tensor, shape: tuple[int, ...]) -> Tensor:
    """
    Convert flattened indices into multidimensional grid indices.

    Args:
        flat_ids: :math:`(m)`, Flattened indices.
        shape: :math:`d`, Grid sizes along each dimension.

    Returns:
        :math:`(m, d)`, Multidimensional indices.
    """
    if is_multidim(shape):
        return torch.stack(torch.unravel_index(flat_ids, shape), dim=-1)
    else:
        return flat_ids


def scatter(data: Tensor, flat_ids: Tensor, shape: int | tuple[int, ...]) -> Tensor:
    r"""
    Expand sequentially packed values into a zero-filled multidimensional grid.

    Args:
        data: :math:`(\dots, m)`, Sequentially packed data.
        flat_ids: :math:`(m)`, Flattened indices.
        shape: :math:`d`, Grid sizes along each dimension.

    Returns:
        :math:`(\dots, n_1, \dots, n_d)`, Zero-filled multidimensional data.
    """
    grid_shape = shape if isinstance(shape, tuple) else (shape,)
    (*prefix_shape, _) = data.shape
    out_shape = (*prefix_shape, *grid_shape)
    num_grid_elements = math.prod(grid_shape)
    out = data.new_zeros((*prefix_shape, num_grid_elements))
    out[..., flat_ids] = data
    return out.view(out_shape)


def gather(data: Tensor, flat_ids: Tensor, shape: int | tuple[int, ...]) -> Tensor:
    r"""
    Collect sequentially packed values from a multidimensional grid.

    Args:
        data: :math:`(\dots, n_1, \dots, n_d)`, Multidimensional data.
        flat_ids: :math:`(m)`, Flattened indices.
        shape: :math:`d`, Grid sizes along each dimension.

    Returns:
        :math:`(\dots, m)`, Sequentially packed data.
    """
    grid_ndim = len(shape) if isinstance(shape, tuple) else 1
    start_dim = data.ndim - grid_ndim
    return data.flatten(start_dim).index_select(dim=-1, index=flat_ids)
