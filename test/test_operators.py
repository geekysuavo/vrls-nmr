import cmath
import math

import pytest
import torch
from torch import Tensor

import vrlsnmr


@pytest.fixture(params=("10x100", "50x100", "20x80"))
def shape(request) -> tuple[int, int]:
    return tuple(map(int, request.param.split("x")))


@pytest.fixture(params=("cpu", "cuda"))
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(params=(1.0e2, 1.0e4))
def tau(request) -> float:
    return request.param


@pytest.fixture
def matrices(shape, device) -> tuple[Tensor, ...]:
    (m, n) = shape

    ids = torch.randperm(n).narrow(dim=0, start=0, length=m).sort(dim=0).values
    B = torch.eye(n).index_select(dim=0, index=ids)

    omega = cmath.exp(-2j * math.pi / n)
    i = torch.arange(n).unsqueeze(dim=0)
    j = torch.arange(n).unsqueeze(dim=1)
    Phi = omega**(i * j) / math.sqrt(n)
    Phi = Phi.t().conj()

    A = B.cfloat() @ Phi

    A = A.to(device)
    B = B.to(device)
    Phi = Phi.to(device)
    ids = ids.to(device)

    return (A, B, Phi, ids)


def test_kernel(matrices, tau):
    (A, B, Phi, ids) = matrices
    (m, n) = A.shape
    device = A.device

    tau = torch.tensor(tau, device=device)
    w = torch.randn((n, 10), device=device).square().mean(dim=1) * 1.0e3

    Winv = w.reciprocal().diag().cfloat()
    Kref = torch.eye(m, device=device) / tau + A @ Winv @ A.t().conj()

    K = vrlsnmr.kernel(w, ids, tau)

    torch.testing.assert_close(Kref, K)


def test_xmarginal(matrices, tau):
    (A, B, Phi, ids) = matrices
    (m, n) = A.shape
    device = A.device

    w = torch.randn((n, 10), device=device).square().mean(dim=1) * 1.0e3
    Gamma_inv = w.diag() + tau * A.t().conj() @ A
    gref = torch.linalg.inv(Gamma_inv).real.diag()

    Winv = w.reciprocal().diag().cfloat()
    Kref = torch.eye(m, device=device) / tau + A @ Winv @ A.t().conj()
    Kinv = torch.linalg.inv(Kref)

    g = vrlsnmr.xmarginal(Kinv, w, ids)

    torch.testing.assert_close(gref, g)


def test_ymarginal(matrices, tau):
    (A, B, Phi, ids) = matrices
    (m, n) = A.shape
    device = A.device

    w = torch.randn((n, 10), device=device).square().mean(dim=1) * 1.0e3
    Gamma_inv = w.diag() + tau * A.t().conj() @ A
    Gamma = torch.linalg.inv(Gamma_inv)
    Sigma = Phi @ Gamma @ Phi.t().conj()
    sref = Sigma.real.diag()

    Winv = w.reciprocal().diag().cfloat()
    Kref = torch.eye(m, device=device) / tau + A @ Winv @ A.t().conj()
    Kinv = torch.linalg.inv(Kref)

    s = vrlsnmr.ymarginal(Kinv, w, ids)

    torch.testing.assert_close(sref, s)
