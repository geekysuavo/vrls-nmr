import cmath
import math

import pytest
import torch

from vrlsnmr.algorithms import vrls, vrls_ex, vrls_mf


def test_vrls():
    (bs, k, m, n) = (4, 10, 50, 100)
    stdev = 0.01
    tau = 1 / stdev**2
    xi = tau

    ids = torch.randperm(n).narrow(dim=0, start=0, length=m).sort(dim=0).values
    B = torch.eye(n).index_select(dim=0, index=ids)

    omega = cmath.exp(-2j * math.pi / n)
    i = torch.arange(n).unsqueeze(dim=0)
    j = torch.arange(n).unsqueeze(dim=1)
    Phi = omega**(i * j) / math.sqrt(n)
    Phi = Phi.t().conj()  # idft
    A = B.cfloat() @ Phi

    x0 = torch.zeros(bs, n)
    for b in range(bs):
        x0_ids = torch.randperm(n).narrow(dim=0, start=0, length=k)
        x0[b, x0_ids] = 1.0

    x0 = x0.cfloat()

    noise = stdev * torch.randn((bs, m), dtype=torch.cfloat)
    y = A.view(1, m, n) @ x0.view(bs, n, 1) + noise.view(bs, m, 1)
    y = y.squeeze(dim=2)
    assert y.shape == (bs, m)

    out_cpu = vrls(y, ids, tau, xi, n, 10)

    y = y.cuda()
    ids = ids.cuda()

    out_cuda = vrls(y, ids, tau, xi, n, 10)

    assert len(out_cpu) == len(out_cuda)
    for x_cpu, x_cuda in zip(out_cpu, out_cuda):
        torch.testing.assert_close(x_cpu, x_cuda.cpu())

    xmean = out_cpu[0]
    assert xmean.abs().gt(0.5).sum(dim=1).eq(k).all()


def test_vrls_ex():
    (bs, k, m, n) = (4, 10, 50, 100)
    stdev = 0.01
    beta_tau = 1e6
    beta_xi = 1e6

    ids = torch.randperm(n).narrow(dim=0, start=0, length=m).sort(dim=0).values
    B = torch.eye(n).index_select(dim=0, index=ids)

    omega = cmath.exp(-2j * math.pi / n)
    i = torch.arange(n).unsqueeze(dim=0)
    j = torch.arange(n).unsqueeze(dim=1)
    Phi = omega**(i * j) / math.sqrt(n)
    Phi = Phi.t().conj()  # idft
    A = B.cfloat() @ Phi

    x0 = torch.zeros(bs, n)
    for b in range(bs):
        x0_ids = torch.randperm(n).narrow(dim=0, start=0, length=k)
        x0[b, x0_ids] = 1.0

    x0 = x0.cfloat()

    noise = stdev * torch.randn((bs, m), dtype=torch.cfloat)
    y = A.view(1, m, n) @ x0.view(bs, n, 1) + noise.view(bs, m, 1)
    y = y.squeeze(dim=2)
    assert y.shape == (bs, m)

    out_cpu = vrls_ex(y, ids, beta_tau, beta_xi, n, 10)

    y = y.cuda()
    ids = ids.cuda()

    out_cuda = vrls_ex(y, ids, beta_tau, beta_xi, n, 10)

    assert len(out_cpu) == len(out_cuda)
    for x_cpu, x_cuda in zip(out_cpu, out_cuda):
        torch.testing.assert_close(x_cpu, x_cuda.cpu())

    xmean = out_cpu[0]
    assert xmean.abs().gt(0.5).sum(dim=1).eq(k).all()


def test_vrls_mf():
    (bs, k, m, n) = (4, 10, 50, 100)
    stdev = 0.01
    tau = 1 / stdev**2
    xi = tau

    ids = torch.randperm(n).narrow(dim=0, start=0, length=m).sort(dim=0).values
    B = torch.eye(n).index_select(dim=0, index=ids)

    omega = cmath.exp(-2j * math.pi / n)
    i = torch.arange(n).unsqueeze(dim=0)
    j = torch.arange(n).unsqueeze(dim=1)
    Phi = omega**(i * j) / math.sqrt(n)
    Phi = Phi.t().conj()  # idft
    A = B.cfloat() @ Phi

    x0 = torch.zeros(bs, n)
    for b in range(bs):
        x0_ids = torch.randperm(n).narrow(dim=0, start=0, length=k)
        x0[b, x0_ids] = 1.0

    x0 = x0.cfloat()

    noise = stdev * torch.randn((bs, m), dtype=torch.cfloat)
    y = A.view(1, m, n) @ x0.view(bs, n, 1) + noise.view(bs, m, 1)
    y = y.squeeze(dim=2)
    assert y.shape == (bs, m)

    out_cpu = vrls_mf(y, ids, tau, xi, n, 100)

    y = y.cuda()
    ids = ids.cuda()

    out_cuda = vrls_mf(y, ids, tau, xi, n, 100)

    assert len(out_cpu) == len(out_cuda)
    for x_cpu, x_cuda in zip(out_cpu, out_cuda):
        torch.testing.assert_close(x_cpu, x_cuda.cpu())

    xmean = out_cpu[0]
    assert xmean.abs().gt(0.5).sum(dim=1).eq(k).all()
