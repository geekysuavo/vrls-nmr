import cmath
import math

import pytest
import torch

from vrlsnmr.algorithms import vrls


def test_vrls():
    (k, m, n) = (10, 50, 200)
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

    x0 = torch.zeros(n)
    x0_ids = torch.randperm(n).narrow(dim=0, start=0, length=k)
    x0[x0_ids] = 1.0
    x0 = x0.cfloat()

    noise = stdev * torch.randn(m, dtype=torch.cfloat)
    y = A @ x0 + noise

    (xmean_cpu, xvar_cpu) = vrls(y, ids, tau, xi, n, 100)

    y = y.cuda()
    ids = ids.cuda()

    (xmean_cuda, xvar_cuda) = vrls(y, ids, tau, xi, n, 100)

    torch.testing.assert_close(xmean_cpu, xmean_cuda.cpu())
    torch.testing.assert_close(xvar_cpu, xvar_cuda.cpu())

    assert xmean_cpu.abs().gt(0.5).sum().item() == k
