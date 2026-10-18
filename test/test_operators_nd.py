# Standard
import functools
import math

# Third party
import pytest
import torch
from torch import Tensor

# VRLS
import vrlsnmr.operators as op


def make_idft_matrix(
    channel: int,
    grid: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    matrices = []
    for dim, grid_size in enumerate(grid):
        k = torch.arange(grid_size, device=device, dtype=dtype)
        matrix = (2j * math.pi * k.outer(k) / grid_size).exp() / math.sqrt(grid_size)
        if dim >= 1 and (channel >> (dim - 1)) & 1:
            matrix = matrix.conj()

        matrices.append(matrix)

    return functools.reduce(torch.kron, matrices)


@pytest.fixture(params=("16", "8x12", "4x8x6"))
def grid(request) -> tuple[int, ...]:
    return tuple(map(int, request.param.split("x")))


@pytest.fixture(params=("cpu", "cuda"))
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(params=("float", "double"))
def dtype(request) -> torch.dtype:
    return getattr(torch, request.param)


@pytest.fixture(params=(1.0e2, 1.0e4))
def tau(request) -> float:
    return request.param


@pytest.fixture
def tensors(device, dtype, grid, tau) -> tuple[Tensor | tuple[Tensor, ...], ...]:
    complex_dtype = dtype.to_complex()
    nch = 1 << (len(grid) - 1)
    n_grid = math.prod(grid)
    m = n_grid // 4
    bs = 4

    flat_ids = (
        torch.randperm(n_grid).narrow(dim=0, start=0, length=m).sort(dim=0).values
    )
    ids = torch.stack(torch.unravel_index(flat_ids, grid), dim=-1)
    ids = ids.to(device=device, dtype=torch.long)

    tau = (
        torch.randn((bs, 50), device=device, dtype=dtype).square().mean(dim=-1).mul(tau)
    )

    weights = (
        torch.randn((bs, *grid, 10), device=device, dtype=dtype)
        .square()
        .mean(dim=-1)
        .mul(1.0e3)
    )
    flat_weights = weights.view(bs, n_grid)

    kernels = []
    xmarginals = []
    ymarginals = []
    for channel in range(nch):
        idft_matrix = make_idft_matrix(channel, grid, device, dtype)
        meas_matrix = idft_matrix[flat_ids]

        inv_weights = flat_weights.reciprocal().diag_embed().to(complex_dtype)
        kernel = torch.eye(m, device=device, dtype=complex_dtype) / tau.view(-1, 1, 1)
        kernel = kernel + meas_matrix @ inv_weights @ meas_matrix.mH

        gamma = torch.linalg.inv(
            flat_weights.diag_embed().to(complex_dtype)
            + tau.view(-1, 1, 1) * meas_matrix.mH @ meas_matrix
        )
        xmarginal = gamma.real.diagonal(dim1=-2, dim2=-1)

        sigma = idft_matrix @ gamma @ idft_matrix.mH
        ymarginal = sigma.real.diagonal(dim1=-2, dim2=-1)

        kernels.append(kernel)
        xmarginals.append(xmarginal)
        ymarginals.append(ymarginal)

    kernels = torch.stack(kernels, dim=1)
    xmarginals = torch.stack(xmarginals).mean(dim=0).view(bs, *grid)
    ymarginals = torch.stack(ymarginals).mean(dim=0).view(bs, *grid)
    refs = (kernels, xmarginals, ymarginals)

    return (ids, tau, weights, refs)


def test_kernel_nd(tensors):
    (ids, tau, w, refs) = tensors
    (ref_kernel, _, _) = refs

    kernel = op.kernel(w, tau, ids)

    torch.testing.assert_close(kernel, ref_kernel, rtol=1e-6, atol=1e-6)
    assert kernel.diagonal(dim1=2, dim2=3).real.gt(0).all()
    assert kernel.diagonal(dim1=2, dim2=3).imag.eq(0).all()


def test_xmarginal_nd(tensors):
    (ids, _, w, refs) = tensors
    (ref_kernel, ref_xmarginal, _) = refs

    kernel_inv = torch.linalg.inv(ref_kernel)
    xmarginal = op.xmarginal(kernel_inv, w, ids)

    torch.testing.assert_close(xmarginal, ref_xmarginal, rtol=1e-6, atol=1e-6)
    assert xmarginal.gt(0).all()


def test_ymarginal_nd(tensors):
    (ids, _, w, refs) = tensors
    (ref_kernel, _, ref_ymarginal) = refs

    kernel_inv = torch.linalg.inv(ref_kernel)
    ymarginal = op.ymarginal(kernel_inv, w, ids)

    torch.testing.assert_close(ymarginal, ref_ymarginal, rtol=1e-6, atol=1e-6)
    assert ymarginal.gt(0).all()
