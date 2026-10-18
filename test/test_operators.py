# Standard
import math

# Third party
import pytest
import torch
from torch import Tensor

# VRLS
import vrlsnmr.operators as op


@pytest.fixture(params=("10x100", "50x100", "20x80"))
def shape(request) -> tuple[int, int]:
    (m, n) = map(int, request.param.split("x"))
    return (m, n)


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
def tensors(device, dtype, shape, tau) -> tuple[Tensor | tuple[Tensor, ...], ...]:
    complex_dtype = dtype.to_complex()
    (m, n) = shape
    bs = 4

    ids = (
        torch.randperm(n, device=device, dtype=torch.long)
        .narrow(dim=0, start=0, length=m)
        .sort(dim=0)
        .values
    )
    ids_matrix = torch.eye(n, device=device, dtype=complex_dtype).index_select(
        dim=0, index=ids
    )

    tau = (
        torch.randn((bs, 50), device=device, dtype=dtype).square().mean(dim=-1).mul(tau)
    )

    weights = (
        torch.randn((bs, n, 10), device=device, dtype=dtype)
        .square()
        .mean(dim=-1)
        .mul(1.0e3)
    )

    k = torch.arange(n, device=device, dtype=dtype)
    idft_matrix = (2j * math.pi * k.outer(k) / n).exp() / math.sqrt(n)
    meas_matrix = ids_matrix @ idft_matrix

    inv_weights = weights.reciprocal().diag_embed().to(complex_dtype)
    kernel = torch.eye(m, device=device, dtype=complex_dtype) / tau.view(-1, 1, 1)
    kernel = kernel + meas_matrix @ inv_weights @ meas_matrix.mH

    gamma_inv = weights.diag_embed() + tau.view(-1, 1, 1) * meas_matrix.mH @ meas_matrix
    gamma = torch.linalg.inv(gamma_inv)
    xmarginal = gamma.real.diagonal(dim1=-2, dim2=-1)

    sigma = idft_matrix @ gamma @ idft_matrix.mH
    ymarginal = sigma.real.diagonal(dim1=-2, dim2=-1)

    refs = (kernel, xmarginal, ymarginal)
    return (ids, tau, weights, refs)


def test_kernel(tensors):
    (ids, tau, w, refs) = tensors
    (ref_kernel, _, _) = refs

    kernel = op.kernel(w, tau, ids)

    torch.testing.assert_close(kernel, ref_kernel, rtol=1e-6, atol=1e-6)
    assert kernel.diagonal(dim1=1, dim2=2).real.gt(0).all()
    assert kernel.diagonal(dim1=1, dim2=2).imag.eq(0).all()


def test_xmarginal(tensors):
    (ids, _, w, refs) = tensors
    (ref_kernel, ref_xmarginal, _) = refs

    kernel_inv = torch.linalg.inv(ref_kernel)
    xmarginal = op.xmarginal(kernel_inv, w, ids)

    torch.testing.assert_close(xmarginal, ref_xmarginal, rtol=1e-6, atol=1e-6)
    assert xmarginal.gt(0).all()


def test_ymarginal(tensors):
    (ids, _, w, refs) = tensors
    (ref_kernel, _, ref_ymarginal) = refs

    kernel_inv = torch.linalg.inv(ref_kernel)
    ymarginal = op.ymarginal(kernel_inv, w, ids)

    torch.testing.assert_close(ymarginal, ref_ymarginal, rtol=1e-6, atol=1e-6)
    assert ymarginal.gt(0).all()
