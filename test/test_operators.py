import cmath
import itertools
import math

import pytest
import torch
from torch import Tensor

import vrlsnmr.operators as op


@pytest.fixture(params=("10x100", "50x100", "20x80"))
def shape(request) -> tuple[int, int]:
    return tuple(map(int, request.param.split("x")))


@pytest.fixture(params=("cpu", "cuda"))
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(params=("float", "double"))
def dtype(request) -> torch.dtype:
    return getattr(torch, request.param)


@pytest.fixture(params=("int", "long"))
def int_dtype(request) -> torch.dtype:
    return getattr(torch, request.param)


@pytest.fixture(params=(1.0e2, 1.0e4))
def tau(request) -> float:
    return request.param


@pytest.fixture
def matrices(tau, shape, device, dtype) -> tuple[Tensor, ...]:
    (m, n) = shape
    bs = 4

    ids = torch.randperm(n).narrow(dim=0, start=0, length=m).sort(dim=0).values
    B = torch.eye(n).index_select(dim=0, index=ids)

    omega = cmath.exp(-2j * math.pi / n)
    i = torch.arange(n).unsqueeze(dim=0)
    j = torch.arange(n).unsqueeze(dim=1)
    Phi = omega**(i * j) / math.sqrt(n)
    Phi = Phi.t().conj()

    A = B.cfloat() @ Phi

    A = A.to(device=device, dtype=dtype.to_complex())
    B = B.to(device=device, dtype=dtype)
    Phi = Phi.to(device=device, dtype=dtype.to_complex())
    ids = ids.to(device=device, dtype=torch.long)

    tau = (
        torch.randn((bs, 50), device=device, dtype=dtype)
        .square()
        .mean(dim=-1)
        .mul(tau)
    )

    weights = (
        torch.randn((bs, n, 10), device=device, dtype=dtype)
        .square()
        .mean(dim=-1)
        .mul(1.0e3)
    )

    return (A, B, Phi, ids, tau, weights)


def test_kernel(matrices):
    (A, B, Phi, ids, tau, w) = matrices
    (m, n) = A.shape

    Im = torch.eye(m, device=A.device, dtype=A.dtype)
    Winv = w.reciprocal().diag_embed().to(dtype=A.dtype)
    Kref = Im / tau.view(-1, 1, 1) + A @ Winv @ A.t().conj()

    K = op.kernel(w, tau, ids)

    torch.testing.assert_close(Kref, K, rtol=1e-6, atol=1e-6)
    assert K.diagonal(dim1=1, dim2=2).real.gt(0).all()
    assert K.diagonal(dim1=1, dim2=2).imag.eq(0).all()


def test_xmarginal(matrices):
    (A, B, Phi, ids, tau, w) = matrices
    (m, n) = A.shape

    Im = torch.eye(m, device=A.device, dtype=A.dtype)
    Winv = w.reciprocal().diag_embed().to(dtype=A.dtype)
    Gamma_inv = w.diag_embed() + tau.view(-1, 1, 1) * A.t().conj() @ A
    gref = torch.linalg.inv(Gamma_inv).real.diagonal(dim1=-2, dim2=-1)
    Kref = Im / tau.view(-1, 1, 1) + A @ Winv @ A.t().conj()
    Kinv = torch.linalg.inv(Kref)

    g = op.xmarginal(Kinv, w, ids)

    torch.testing.assert_close(gref, g, rtol=1e-6, atol=1e-6)
    assert g.gt(0).all()


def test_ymarginal(matrices):
    (A, B, Phi, ids, tau, w) = matrices
    (m, n) = A.shape

    Im = torch.eye(m, device=A.device, dtype=A.dtype)
    Winv = w.reciprocal().diag_embed().to(dtype=A.dtype)
    Gamma_inv = w.diag_embed() + tau.view(-1, 1, 1) * A.t().conj() @ A
    Gamma = torch.linalg.inv(Gamma_inv)
    Sigma = Phi @ Gamma @ Phi.t().conj()
    sref = Sigma.real.diagonal(dim1=-2, dim2=-1)
    Kref = Im / tau.view(-1, 1, 1) + A @ Winv @ A.t().conj()
    Kinv = torch.linalg.inv(Kref)

    s = op.ymarginal(Kinv, w, ids)

    torch.testing.assert_close(sref, s, rtol=1e-6, atol=1e-6)
    assert s.gt(0).all()


@pytest.mark.parametrize(
    ("m", "n"),
    itertools.chain(
        itertools.combinations(range(2, 64, 4), 2),
        itertools.combinations(range(64, 256, 16), 2),
        itertools.combinations(range(256, 1024, 64), 2),
    ),
)
def test_schedexp(int_dtype, m, n):
    rate = -math.log(0.2) / n
    ids = op.schedexp(rate, m, n, dtype=int_dtype)

    assert ids.dtype == int_dtype
    assert ids.shape == (m,)
    assert ids.ge(0).all()
    assert ids.lt(n).all()
    assert ids.unique().numel() == m


@pytest.mark.parametrize(
    ("m", "n"),
    itertools.chain(
        itertools.combinations(range(2, 64, 4), 2),
        itertools.combinations(range(64, 256, 16), 2),
        itertools.combinations(range(256, 1024, 64), 2),
    ),
)
def test_schedpg(int_dtype, m, n):
    ids = op.schedpg(m, n, dtype=int_dtype)

    assert ids.dtype == int_dtype
    assert ids.shape == (m,)
    assert ids.ge(0).all()
    assert ids.lt(n).all()
    assert ids.unique().numel() == m
