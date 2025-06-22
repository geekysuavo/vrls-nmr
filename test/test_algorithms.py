import cmath
import math

import pytest
import torch

from vrlsnmr.algorithms import vrls


def test_vrls():
    (bs, k, m, n) = (4, 10, 50, 200)
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

    (xmean_cpu, xvar_cpu) = vrls(y, ids, tau, xi, n, 100)

    y = y.cuda()
    ids = ids.cuda()

    (xmean_cuda, xvar_cuda) = vrls(y, ids, tau, xi, n, 100)

    torch.testing.assert_close(xmean_cpu, xmean_cuda.cpu())
    torch.testing.assert_close(xvar_cpu, xvar_cuda.cpu())

    assert xmean_cpu.abs().gt(0.5).sum(dim=1).eq(k).all()
