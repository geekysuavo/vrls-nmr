# Third party
import pytest
import torch

# VRLS
import vrlsnmr.multicomplex as mx


def randn_complex(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, 2).view(torch.cfloat).squeeze(dim=-1)


@pytest.fixture(params=(2, 8))
def batch_size(request) -> int:
    return request.param


@pytest.fixture(params=("6", "4x8", "3x4x5", "3x2x4x7"))
def grid_shape(request) -> tuple[int, ...]:
    return tuple(map(int, request.param.split("x")))


@pytest.fixture
def shape(batch_size, grid_shape) -> tuple[int, ...]:
    num_channels = 1 << (len(grid_shape) - 1)
    return (batch_size, num_channels, *grid_shape)


def test_format_conversion(shape):
    y = randn_complex(*shape)
    z = mx.to_complex(y)

    y_new = mx.to_states(z)
    z_new = mx.to_complex(y_new)

    y_norm = y.abs().square().sum()
    z_norm = y.abs().square().sum()

    torch.testing.assert_close(y, y_new)
    torch.testing.assert_close(z, z_new)
    torch.testing.assert_close(y_norm, z_norm)


@pytest.mark.parametrize("grid_size", [8, 16])
def test_fftn_1d(batch_size, grid_size):
    y = randn_complex(batch_size, grid_size)
    y_fft = mx.fftn(y, inverse=False)
    y_new = mx.fftn(y_fft, inverse=True)

    torch.testing.assert_close(y, y_new)
    assert y_fft.shape == y.shape


@pytest.mark.parametrize("states", [False, True])
def test_fftn_nd(shape, states):
    y = randn_complex(*shape)
    y_fft = mx.fftn(y, inverse=False, states=states)
    y_new = mx.fftn(y_fft, inverse=True, states=states)

    torch.testing.assert_close(y, y_new)
    assert y_fft.shape == y.shape
