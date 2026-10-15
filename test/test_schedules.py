# Standard
import math

# Third party
import pytest
import torch

# VRLS
import vrlsnmr.multidim as md
import vrlsnmr.schedules as sch


@pytest.fixture(params=("10x100", "50x200", "64x16x16", "128x16x32x32"))
def config(request) -> tuple[int, int | tuple[int, ...]]:
    (m, *n) = map(int, request.param.split("x"))
    if len(n) == 1:
        return (m, n[0])
    else:
        return (m, tuple(n))


@pytest.fixture(params=("cpu", "cuda"))
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(params=("int", "long"))
def dtype(request) -> torch.dtype:
    return getattr(torch, request.param)


def test_uniform(config, dtype, device):
    (m, n) = config
    ids = sch.uniform(m, n, dtype=dtype, device=device)

    shape = (m,) if isinstance(n, int) else (m, len(n))
    nmax = torch.tensor(n, device=device)

    assert ids.dtype == dtype
    assert ids.shape == shape
    assert ids.ge(0).all()
    assert ids.lt(nmax).all()
    assert md.ravel_ids(ids, n).unique().numel() == m


def test_exponential(config, dtype, device):
    (m, n) = config
    rate = -math.log(0.2) / (n if isinstance(n, int) else math.prod(n))
    ids = sch.exponential(rate, m, n, dtype=dtype, device=device)

    shape = (m,) if isinstance(n, int) else (m, len(n))
    nmax = torch.tensor(n, device=device)

    assert ids.dtype == dtype
    assert ids.shape == shape
    assert ids.ge(0).all()
    assert ids.lt(nmax).all()
    assert md.ravel_ids(ids, n).unique().numel() == m


def test_poisson_gap(config, dtype, device):
    (m, n) = config
    ids = sch.poisson_gap(m, n, dtype=dtype, device=device)

    shape = (m,) if isinstance(n, int) else (m, len(n))
    nmax = torch.tensor(n, device=device)

    assert ids.dtype == dtype
    assert ids.shape == shape
    assert ids.ge(0).all()
    assert ids.lt(nmax).all()
    assert md.ravel_ids(ids, n).unique().numel() == m
