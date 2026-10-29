# Standard
import math

# Third party
import pytest
import torch
from torch import Tensor

# VRLS
from vrlsnmr.algorithms import ists, vrls, vrls_ex, vrls_mf
from vrlsnmr.simulators import Signal


def build_instance(
    *,
    bs: int = 4,
    k: int = 10,
    m: int = 50,
    n: tuple[int, ...] = (),
    stdev: float = 0.01,
) -> tuple[Tensor, Tensor]:
    assert n
    ndim = len(n)
    n_grid = math.prod(n)
    ids = torch.randperm(n_grid).narrow(dim=0, start=0, length=m)
    ids = torch.stack(torch.unravel_index(ids, n), dim=-1)

    peaks = torch.multinomial(torch.ones(bs, n_grid), num_samples=k)
    peaks = torch.stack(torch.unravel_index(peaks, n), dim=-1)
    frequencies = 0.5 * peaks / torch.tensor(n)

    signal = Signal(
        frequencies=frequencies,
        amplitudes=torch.ones(bs, k) / math.sqrt(n_grid),
        decayrates=torch.zeros(bs, k, ndim),
        phases=torch.zeros(bs, k, ndim),
    )

    y = signal(ids, noise=stdev / math.sqrt(n_grid))
    return (y, ids)


@pytest.mark.parametrize("split", [False, True])
def test_vrls(split):
    (k, m, n) = (10, 90, (10, 10))
    stdev = 0.01
    niter = 10
    tau = 1 / stdev**2
    xi = tau

    (y, ids) = build_instance(k=k, m=m, n=n, stdev=stdev)

    nout = (20, 20)
    if split:
        (*_, w) = vrls(y, ids, tau, xi, nout, niter // 2, warm_start=True)
        (*out_cpu, _) = vrls(y, ids, tau, xi, nout, niter // 2, warm_start=w)
    else:
        out_cpu = vrls(y, ids, tau, xi, nout, niter)

    y = y.cuda()
    ids = ids.cuda()

    if split:
        (*_, w) = vrls(y, ids, tau, xi, nout, niter // 2, warm_start=True)
        (*out_cuda, _) = vrls(y, ids, tau, xi, nout, niter // 2, warm_start=w)
    else:
        out_cuda = vrls(y, ids, tau, xi, nout, niter)

    assert len(out_cpu) == len(out_cuda)
    for x_cpu, x_cuda in zip(out_cpu, out_cuda):
        torch.testing.assert_close(x_cpu, x_cuda.cpu())

    xmean = out_cpu[0]
    xmean_abs = xmean.abs().square().sum(dim=1).sqrt()
    xmean_abs = xmean_abs[:, :10, :10]

    npeaks = xmean_abs.gt(0.5).flatten(start_dim=1).sum(dim=1)
    assert npeaks.ge(k - 1).all()
    assert npeaks.le(k + 1).all()


@pytest.mark.parametrize("split", [False, True])
def test_vrls_ex(split):
    (k, m, n) = (10, 90, (10, 10))
    stdev = 0.01
    niter = 10
    tau = 1.0e9
    xi = tau

    (y, ids) = build_instance(k=k, m=m, n=n, stdev=stdev)

    nout = (20, 20)
    if split:
        (*_, w) = vrls_ex(y, ids, tau, xi, nout, niter // 2, warm_start=True)
        (*out_cpu, _) = vrls_ex(y, ids, tau, xi, nout, niter // 2, warm_start=w)
    else:
        out_cpu = vrls_ex(y, ids, tau, xi, nout, niter)

    y = y.cuda()
    ids = ids.cuda()

    if split:
        (*_, w) = vrls_ex(y, ids, tau, xi, nout, niter // 2, warm_start=True)
        (*out_cuda, _) = vrls_ex(y, ids, tau, xi, nout, niter // 2, warm_start=w)
    else:
        out_cuda = vrls_ex(y, ids, tau, xi, nout, niter)

    assert len(out_cpu) == len(out_cuda)
    for x_cpu, x_cuda in zip(out_cpu, out_cuda):
        torch.testing.assert_close(x_cpu, x_cuda.cpu())

    xmean = out_cpu[0]
    xmean_abs = xmean.abs().square().sum(dim=1).sqrt()
    xmean_abs = xmean_abs[:, :10, :10]

    npeaks = xmean_abs.gt(0.5).flatten(start_dim=1).sum(dim=1)
    assert npeaks.ge(k - 2).all()
    assert npeaks.le(k + 2).all()


@pytest.mark.parametrize("split", [False, True])
def test_vrls_mf(split):
    (k, m, n) = (10, 1250, (50, 50))
    stdev = 0.01
    niter = 100
    tau = 1 / stdev**2
    xi = tau

    (y, ids) = build_instance(k=k, m=m, n=n, stdev=stdev)

    nout = (100, 100)
    if split:
        (*_, w) = vrls_mf(y, ids, tau, xi, nout, niter // 2, warm_start=True)
        (*out_cpu, _) = vrls_mf(y, ids, tau, xi, nout, niter // 2, warm_start=w)
    else:
        out_cpu = vrls_mf(y, ids, tau, xi, nout, niter)

    y = y.cuda()
    ids = ids.cuda()

    if split:
        (*_, w) = vrls_mf(y, ids, tau, xi, nout, niter // 2, warm_start=True)
        (*out_cuda, _) = vrls_mf(y, ids, tau, xi, nout, niter // 2, warm_start=w)
    else:
        out_cuda = vrls_mf(y, ids, tau, xi, nout, niter)

    assert len(out_cpu) == len(out_cuda)
    for x_cpu, x_cuda in zip(out_cpu, out_cuda):
        torch.testing.assert_close(x_cpu, x_cuda.cpu())

    xmean = out_cpu[0]
    xmean_abs = xmean.abs().square().sum(dim=1).sqrt()
    xmean_abs = xmean_abs[:, :50, :50]

    assert xmean_abs.gt(0.5).flatten(start_dim=1).sum(dim=1).eq(k).all()


def test_ists():
    (k, m, n) = (10, 250, (50, 50))
    stdev = 0.001
    niter = 200
    mu = 0.98

    (y, ids) = build_instance(k=k, m=m, n=n, stdev=stdev)

    nout = (100, 100)
    out_cpu = ists(y, ids, mu, nout, niter)

    y = y.cuda()
    ids = ids.cuda()

    out_cuda = ists(y, ids, mu, nout, niter)

    assert len(out_cpu) == len(out_cuda)
    for x_cpu, x_cuda in zip(out_cpu, out_cuda):
        torch.testing.assert_close(x_cpu, x_cuda.cpu())

    xhat = out_cpu[0]
    xhat_abs = xhat.abs().square().sum(dim=1).sqrt()
    xhat_abs = xhat_abs[:, :50, :50]

    assert xhat_abs.gt(0.5).flatten(start_dim=1).sum(dim=1).eq(k).all()
