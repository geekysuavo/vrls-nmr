# Standard
import math
import random

# Third party
import pytest

# VRLS
import vrlsnmr.gridsizes as gs


@pytest.mark.parametrize(
    ("m", "n_prev", "n_next"),
    [
        (1, 5, 10),
        (1, 10, 10),
        (8, 30, 60),
        (8, 60, 60),
        (128, 256, 1024),
        (128, 800, 1024),
    ],
)
def test_doubled(m, n_prev, n_next):
    adapt_size = gs.Doubled(n_final=1024, min_sparsity=0.1)
    assert adapt_size(m, n_prev) == n_next


@pytest.mark.parametrize(
    ("m", "n_prev", "n_next"),
    [
        (1, 5, 10),
        (1, 10, 10),
        (8, 30, 80),
        (8, 60, 80),
        (128, 256, 1024),
        (128, 800, 1024),
    ],
)
def test_scaled(m, n_prev, n_next):
    adapt_size = gs.Scaled(n_final=1024, min_sparsity=0.1)
    assert adapt_size(m, n_prev) == n_next


@pytest.mark.parametrize(
    ("m", "n_prev", "n_next"),
    [
        pytest.param(10, (2, 2), (4, 4), id="10-5x5-5x5"),
        pytest.param(10, (4, 5), (4, 5), id="10-4x5-4x5"),
        pytest.param(10, (10, 5), (10, 5), id="10-10x5-10x5"),
        pytest.param(10, (10, 10), (10, 10), id="10-10x10-10x10"),
        pytest.param(20, (10, 10), (10, 10), id="20-10x10-10x10"),
        pytest.param(20, (10, 20), (10, 20), id="20-10x20-10x20"),
        pytest.param(20, (10, 2), (10, 8), id="20-10x2-10x8"),
        pytest.param(40, (10, 10), (10, 20), id="40-10x10-10x20"),
    ],
)
def test_doubled_all(m, n_prev, n_next):
    adapt_size = gs.DoubledAll(n_final=(10, 20), min_sparsity=0.2)
    assert adapt_size(m, n_prev) == n_next


@pytest.mark.parametrize(
    ("m", "n_prev", "n_next"),
    [
        pytest.param(10, (5, 5), (10, 5), id="10-5x5-10x5"),
        pytest.param(10, (4, 5), (8, 5), id="10-4x5-8x5"),
        pytest.param(10, (10, 5), (10, 5), id="10-10x5-10x5"),
        pytest.param(10, (10, 10), (10, 10), id="10-10x10-10x10"),
        pytest.param(20, (10, 10), (10, 10), id="20-10x10-10x10"),
        pytest.param(20, (10, 20), (10, 20), id="20-10x20-10x20"),
        pytest.param(20, (10, 2), (10, 8), id="20-10x2-10x8"),
        pytest.param(40, (10, 10), (10, 20), id="40-10x10-10x20"),
    ],
)
def test_doubled_small(m, n_prev, n_next):
    adapt_size = gs.DoubledSmall(n_final=(10, 20), min_sparsity=0.2)
    assert adapt_size(m, n_prev) == n_next


@pytest.mark.parametrize(
    ("m", "n_prev", "n_next"),
    [
        pytest.param(10, (5, 5), (5, 10), id="10-5x5-5x10"),
        pytest.param(10, (4, 5), (4, 12), id="10-4x5-4x12"),
        pytest.param(10, (10, 5), (10, 5), id="10-10x5-10x5"),
        pytest.param(10, (10, 10), (10, 10), id="10-10x10-10x10"),
        pytest.param(20, (10, 10), (10, 10), id="20-10x10-10x10"),
        pytest.param(20, (10, 20), (10, 20), id="20-10x20-10x20"),
        pytest.param(20, (10, 2), (10, 10), id="20-10x2-10x10"),
        pytest.param(40, (10, 10), (10, 20), id="40-10x10-10x20"),
    ],
)
def test_scaled_all(m, n_prev, n_next):
    adapt_size = gs.ScaledAll(n_final=(10, 20), min_sparsity=0.2)
    assert adapt_size(m, n_prev) == n_next


@pytest.mark.parametrize("repeat", list(range(10)))
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("cls", [gs.DoubledAll, gs.DoubledSmall, gs.ScaledAll])
def test_output_properties(cls, ndim, repeat):
    sizes = (2, 4, 8, 12, 16, 24, 32, 64)

    rng = random.Random(repeat * 3 + 1)
    min_sparsity = rng.choice([0.01, 0.05, 0.2])
    n_final = tuple(rng.choice(sizes[2:]) for _ in range(ndim))
    n_prev = tuple(rng.choice([n for n in sizes if n <= nf]) for nf in n_final)
    m = rng.randint(1, 2560)

    adapt = cls(n_final, min_sparsity)
    n_next = adapt(m, n_prev)

    assert all(np <= n <= nf for np, n, nf in zip(n_prev, n_next, n_final))
    assert all(n % 2 == 0 for np, n in zip(n_prev, n_next) if n != np)
    if m / math.prod(n_prev) < min_sparsity:
        assert n_next == n_prev
    else:
        assert m / math.prod(n_next) >= min_sparsity
