# Standard
from functools import partial

# Third party
import pytest
import torch

# VRLS
from vrlsnmr.algorithms import ans, vrls
from vrlsnmr.gridsizes import Doubled, DoubledSmall
from vrlsnmr.simulators import Signal


@pytest.mark.parametrize(("warm_start", "niter"), [(False, 10), (True, 1)])
def test_ans_1d(warm_start, niter):
    min_sparsity = 0.05
    (m_initial, m_final) = (4, 40)
    (n_initial, n_final) = (16, 512)

    stdev = 0.1
    tau = 1 / stdev**2
    ground_truth = Signal(
        amplitudes=torch.ones(2),
        frequencies=torch.tensor([0.1, 0.3]),
        decayrates=torch.full((2,), 0.001),
        phases=torch.zeros(2),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        ground_truth.cuda()
    else:
        device = None

    steps = []
    ans(
        model=vrls,
        measure=partial(ground_truth, noise=stdev),
        adapt_size=Doubled(n_final, min_sparsity),
        m_initial=m_initial,
        m_final=m_final,
        n_initial=n_initial,
        n_final=n_final,
        warm_start=warm_start,
        callback=lambda state: steps.append(state),
        device=device,
        niter=niter,
        tau=tau,
        xi=tau,
    )
    assert len(steps) == m_final - m_initial

    (y, ids, *_) = steps[-1]
    assert y.shape == (m_final,)
    assert ids.shape == (m_final,)
    assert y.isfinite().all()


@pytest.mark.parametrize(("warm_start", "niter"), [(False, 10), (True, 1)])
def test_ans_nd(warm_start, niter):
    min_sparsity = 0.05
    (m_initial, m_final) = (4, 40)
    (n_initial, n_final) = ((4, 4), (16, 16))

    stdev = 0.1
    tau = 1 / stdev**2
    ground_truth = Signal(
        amplitudes=torch.ones(2),
        frequencies=torch.tensor([[0.1, 0.3], [0.2, 0.15]]),
        decayrates=torch.full((2, 2), 0.001),
        phases=torch.zeros(2, 2),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        ground_truth.cuda()
    else:
        device = None

    steps = []
    ans(
        model=vrls,
        measure=partial(ground_truth, noise=stdev),
        adapt_size=DoubledSmall(n_final, min_sparsity),
        m_initial=m_initial,
        m_final=m_final,
        n_initial=n_initial,
        n_final=n_final,
        warm_start=warm_start,
        callback=lambda state: steps.append(state),
        device=device,
        niter=niter,
        tau=tau,
        xi=tau,
    )
    assert len(steps) == m_final - m_initial

    (y, ids, *_) = steps[-1]
    assert y.shape == (2, m_final)
    assert ids.shape == (m_final, 2)
    assert y.isfinite().all()
