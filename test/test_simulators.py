from functools import partial

import pytest
import torch

from vrlsnmr.simulators import (
    constant,
    random_chisq,
    random_normal,
    random_spaced,
    random_unif,
    Signal,
)


@pytest.fixture(params=(1, 2, 3, 5))
def num_components(request) -> int:
    return request.param


@pytest.fixture(params=(1, 4, 8))
def num_replicas(request) -> int:
    return request.param


@pytest.mark.parametrize("value", (1.1, 2.2))
def test_constant(num_components, num_replicas, value):
    out = constant(num_components, num_replicas, value=value)

    assert torch.is_tensor(out)
    assert out.dtype == torch.float
    assert out.shape == (num_replicas, num_components)
    assert out.isfinite().all()
    assert out.eq(value).all()


@pytest.mark.parametrize("mean", (1.0, 55.5))
@pytest.mark.parametrize("dof", (1, 5, 10))
def test_random_chisq(num_components, num_replicas, dof, mean):
    out = random_chisq(num_components, num_replicas, dof=dof, mean=mean)

    assert torch.is_tensor(out)
    assert out.dtype == torch.float
    assert out.shape == (num_replicas, num_components)
    assert out.isfinite().all()
    assert out.gt(0).all()


@pytest.mark.parametrize("stdev", (1.0, 0.1))
@pytest.mark.parametrize("mean", (0.0, 1.0, -2.5))
def test_random_normal(num_components, num_replicas, mean, stdev):
    out = random_normal(num_components, num_replicas, mean=mean, stdev=stdev)

    assert torch.is_tensor(out)
    assert out.dtype == torch.float
    assert out.shape == (num_replicas, num_components)
    assert out.isfinite().all()


@pytest.mark.parametrize("space", (0.001, 0.01))
@pytest.mark.parametrize(("lower", "upper"), ((0.0, 1.0), (3.3, 4.4)))
def test_random_spaced(num_components, num_replicas, lower, upper, space):
    out = random_spaced(
        num_components, num_replicas, lower=lower, upper=upper, space=space
    )

    assert torch.is_tensor(out)
    assert out.dtype == torch.float
    assert out.shape == (num_replicas, num_components)
    assert out.isfinite().all()
    assert out.ge(lower).all()
    assert out.le(upper).all()

    ij = torch.triu_indices(num_components, num_components, 1)
    (i, j) = ij.unbind(dim=0)

    assert (out[:, i] - out[:, j]).abs().gt(space).all()


@pytest.mark.parametrize(("lower", "upper"), ((0.0, 1.0), (3.3, 4.4)))
def test_random_unif(num_components, num_replicas, lower, upper):
    out = random_unif(num_components, num_replicas, lower=lower, upper=upper)

    assert torch.is_tensor(out)
    assert out.dtype == torch.float
    assert out.shape == (num_replicas, num_components)
    assert out.isfinite().all()
    assert out.ge(lower).all()
    assert out.le(upper).all()


def test_signal(num_components, num_replicas):
    signal = Signal.build(
        num_components=num_components,
        num_replicas=num_replicas,
        frequencies=partial(random_unif, lower=-1.0, upper=1.0),
        decayrates=partial(constant, value=0.01),
        amplitudes=partial(random_chisq, dof=5, mean=1.0),
        phases=partial(random_normal, mean=0.0, stdev=0.001),
    )

    for t in (0, 1, 2, 3, 4, 5, 0.0, 1.1, 2.4, 3.5):
        out = signal(t)
        assert torch.is_tensor(out)
        assert out.dtype == torch.complex64
        assert out.shape == (num_replicas, 1)
        assert out.isfinite().all()

    t = torch.arange(128)
    out = signal(t)
    assert torch.is_tensor(out)
    assert out.dtype == torch.complex64
    assert out.shape == (num_replicas, 128)
    assert out.isfinite().all()
