# Standard
import math

# Third party
import pytest
import torch

# VRLS
import vrlsnmr.multidim as md


@pytest.fixture(params=("1", "3x2"))
def prefix_shape(request) -> tuple[int, ...]:
    return tuple(map(int, request.param.split("x")))


@pytest.fixture(params=("6", "4x8", "3x4x5", "3x2x4x7"))
def grid_shape(request) -> tuple[int, ...]:
    return tuple(map(int, request.param.split("x")))


@pytest.mark.parametrize("grid_size", [4, 12])
def test_index_conversion_1d(grid_size):
    num_values = grid_size // 2

    flat_ids = torch.randperm(grid_size)[:num_values]
    grid_ids = md.unravel_ids(flat_ids, grid_size)

    flat_ids_new = md.ravel_ids(grid_ids, grid_size)
    grid_ids_new = md.unravel_ids(flat_ids_new, grid_size)

    assert torch.equal(flat_ids, flat_ids_new)
    assert torch.equal(grid_ids, grid_ids_new)


def test_index_conversion_nd(grid_shape):
    num_grid_elements = math.prod(grid_shape)
    num_values = num_grid_elements // 2

    flat_ids = torch.randperm(num_grid_elements)[:num_values]
    grid_ids = md.unravel_ids(flat_ids, grid_shape)

    flat_ids_new = md.ravel_ids(grid_ids, grid_shape)
    grid_ids_new = md.unravel_ids(flat_ids_new, grid_shape)

    assert torch.equal(flat_ids, flat_ids_new)
    assert torch.equal(grid_ids, grid_ids_new)


@pytest.mark.parametrize("scalar_shape", [False, True])
def test_scatter_gather(prefix_shape, grid_shape, scalar_shape):
    if scalar_shape and len(grid_shape) == 1:
        shape = grid_shape[0]
    else:
        shape = grid_shape

    num_grid_elements = math.prod(grid_shape)
    num_values = num_grid_elements // 2

    flat_ids = torch.randperm(num_grid_elements)[:num_values]
    dense = torch.randn(*prefix_shape, *grid_shape)

    gather_data = md.gather(dense, flat_ids, shape)
    assert gather_data.shape == (*prefix_shape, num_values)

    scatter_data = md.scatter(gather_data, flat_ids, shape)
    assert scatter_data.shape == (*prefix_shape, *grid_shape)

    gather_new = md.gather(scatter_data, flat_ids, shape)
    scatter_new = md.scatter(gather_new, flat_ids, shape)

    torch.testing.assert_close(gather_data, gather_new)
    torch.testing.assert_close(scatter_data, scatter_new)

    mask = torch.zeros(num_grid_elements, dtype=torch.bool)
    mask[flat_ids] = True
    mask = mask.expand(*prefix_shape, num_grid_elements).view_as(dense)
    assert torch.equal(dense == scatter_data, mask)
