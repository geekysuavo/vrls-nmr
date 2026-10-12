r"""
Multicomplex utility functions for multidimensional NUS NMR algorithms.

Algorithms in :mod:`vrlsnmr` assume a uniformly sampled direct dimension
and :math:`d` nonuniformly sampled indirect dimensions, where measurements
are acquired in full quadrature. This multidimensional quadrature is described
by a commutative multicomplex algebra having the imaginary units
:math:`\{ i_1, \dots, i_d \}`, where each :math:`i_k^2 = -1` and
by commutativity, :math:`i_k i_r = i_r i_k`. A single multicomplex value is
stored as :math:`c = 2^{d-1}` complex numbers in States-Haberkorn-Ruben (or
simply States) convention. Precisely,

.. math::

    y = \sum_{s=1}^c y_s \prod_{k=2}^d i_k^{q_k(s)}

where :math:`y_s` is a complex number using the ordinary imaginary unit :math:`i_1`.
The exponents :math:`q_k(s)` equal zero for cosine-modulated components and one for
sine modulated components, and are computed as the bitwise-and of :math:`s`
and :math:`2^k`, i.e. :math:`q_k(s) = s \wedge 2^k`.

For each :math:`k > 1`, the elements :math:`e_{\pm} = (1 \pm i_1 i_k) / 2`
split the multicomplex algebra into two copies of the complex numbers, within which
:math:`i_k` acts as :math:`\mp i_1`. When applied to every dimension, this turns
the :math:`c` multicomplex components into :math:`c` independent complex "channels".
a full :math:`d`-dimensional multicomplex discrete Fourier transform (DFT) becomes
a simple complex DFT along each channel. Channel :math:`s` is negated along dimension
:math:`k > 1` when bit :math:`k - 2` is set in :math:`s`. The minus two is a result of
the first dimension not requiring a channel and the bitmasks being zero based.

Multidimensional data are stored as :math:`(b, c, n_1, \dots, n_d)` when values are
on a :math:`d`-dimensional grid, and as :math:`(b, c, m)` when values are packed
sequentially.
"""

# Standard
import functools
import math

# Third party
import torch
from torch import Tensor


def to_complex(y: Tensor) -> Tensor:
    r"""
    Convert States-convention multicomplex components into complex channels.

    Args:
        y: :math:`(b, c, n_1, \dots, n_d)`, Multicomplex data.

    Returns:
        :math:`(b, c, n_1, \dots, n_d)`, Complex channel data.

    Note:
        This is a variance-preserving (unitary) transformation.
    """
    return _convert_format(y, inverse=False)


def to_states(z: Tensor) -> Tensor:
    r"""
    Convert complex channels into States-convention multicomplex components.

    Args:
        z: :math:`(b, c, n_1, \dots, n_d)`, Complex channel data.

    Returns:
        :math:`(b, c, n_1, \dots, n_d)`, Multicomplex data.
    """
    return _convert_format(z, inverse=True)


def fftn(data: Tensor, *, inverse: bool = False, states: bool = False) -> Tensor:
    r"""
    Multidimensional fast Fourier transform (FFT).

    Args:
        data: :math:`(b, c, n_1, \dots, n_d)`, Input data.
        inverse: When true, return the inverse transform.
        states: When true, the data are accepted and returned in States format.
            Otherwise, they are accepted in returned in complex channel format.

    Returns:
        :math:`(b, c, n_1, \dots, n_d)`, Transformed data.
    """
    if states:
        return to_states(fftn(to_complex(data), inverse=inverse))

    num_grid_dims = data.ndim - 2
    num_channels = data.size(dim=1)
    assert num_channels == 1 << (num_grid_dims - 1)

    grid_dims = tuple(range(2, data.ndim))
    if inverse:
        out = torch.fft.ifftn(data, dim=grid_dims, norm="ortho")
    else:
        out = torch.fft.fftn(data, dim=grid_dims, norm="ortho")

    if num_grid_dims == 1:
        return out
    else:
        return out[_fftn_flip_index(tuple(data.shape), data.device)]


@functools.cache
def _fftn_flip_index(
    shape: tuple[int, ...],
    device: torch.device,
) -> tuple[slice | Tensor, ...]:
    r"""
    Build a gather index used by :func:`fftn` to bin-reverse flipped dimensions.

    Args:
        shape: :math:`(b, c, n_1, \dots, n_d)`, Shape of the input data tensor.
        device: Device to place the resulting tensor indices onto.

    Returns:
        A :class:`tuple containing :math:`d+2` elements

        - A full slice object.
        - :math:`(c, 1, \dots, 1)`, Channel index tensor.
        - :math:`(c, n_1, 1, \dots, 1)`, First grid index tensor.
        - :math:`(c, 1, n_2 1, \dots, 1)`, Second grid index tensor.
        - And so on...
        - :math:`(c, 1, \dots, 1, n_d)`, Final grid index tensor.
    """
    (_, num_channels, *grid_shape) = shape
    channels = torch.arange(num_channels)

    dim_indices = []
    for dim, grid_size in enumerate(grid_shape):
        indices = torch.arange(grid_size)
        if dim == 0:
            bins = indices.expand(num_channels, grid_size)
        else:
            flipped = ((channels >> (dim - 1)) & 1).bool().unsqueeze(dim=1)
            bins = torch.where(flipped, (-indices) % grid_size, indices)

        dim_index = bins.view(
            num_channels,
            *([1] * dim),
            grid_size,
            *([1] * (len(grid_shape) - dim - 1)),
        )
        dim_indices.append(dim_index.to(device))

    channel_index = channels.view(num_channels, *([1] * len(grid_shape)))
    return (slice(None), channel_index.to(device), *dim_indices)


def _convert_format(data: Tensor, *, inverse: bool) -> Tensor:
    """
    Convert between multicomplex States format and complex channel format.

    This is a utility function called by :func:`to_complex` and :func:`to_states`.
    One-dimensional data having a :math:`c=1` pass through unmodified.
    """

    def butterfly(x: Tensor, axis: int) -> Tensor:
        inv_sqrt2 = 1 / math.sqrt(2)
        (a, b) = x.unbind(dim=axis)
        if inverse:
            # (z+, z-) => (y0, y1)
            y0 = a + b
            y1 = -1j * (a - b)
            return inv_sqrt2 * torch.stack((y0, y1), dim=axis)
        else:
            # (y0, y1) => (z+, z-)
            zp = a + 1j * b
            zm = a - 1j * b
            return inv_sqrt2 * torch.stack((zp, zm), dim=axis)

    num_channels = data.size(dim=1)
    num_channel_dims = num_channels.bit_length() - 1
    assert num_channels == 1 << num_channel_dims

    if not num_channel_dims:
        return data

    channel_shape = tuple([2] * num_channel_dims)
    out = data.unflatten(dim=1, sizes=channel_shape)

    for axis in range(1, num_channel_dims + 1):
        out = butterfly(out, axis)

    return out.flatten(start_dim=1, end_dim=num_channel_dims)
