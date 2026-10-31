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
        pytest.param(10, (5, 5), (10, 10), id="10-5x5-10x10"),
        pytest.param(10, (4, 5), (8, 10), id="10-4x5-8x10"),
        pytest.param(10, (10, 10), (10, 10), id="10-10x10-10x10"),
        pytest.param(20, (10, 10), (10, 20), id="20-10x10-10x20"),
        pytest.param(20, (10, 20), (10, 20), id="20-10x20-10x20"),
        pytest.param(20, (10, 2), (10, 16), id="20-10x2-10x16"),
    ],
)
def test_doubled_all(m, n_prev, n_next):
    adapt_size = gs.DoubledAll(n_final=(10, 20), min_sparsity=0.2)
    assert adapt_size(m, n_prev) == n_next


@pytest.mark.parametrize(
    ("m", "n_prev", "n_next"),
    [
        pytest.param(10, (5, 5), (10, 10), id="10-5x5-10x10"),
        pytest.param(10, (4, 5), (8, 10), id="10-4x5-8x10"),
        pytest.param(10, (10, 5), (10, 10), id="10-10x5-10x10"),
        pytest.param(10, (10, 10), (10, 10), id="10-10x10-10x10"),
        pytest.param(20, (10, 10), (10, 20), id="20-10x10-10x20"),
        pytest.param(20, (10, 20), (10, 20), id="20-10x20-10x20"),
        pytest.param(20, (10, 2), (10, 16), id="20-10x2-10x16"),
    ],
)
def test_doubled_small(m, n_prev, n_next):
    adapt_size = gs.DoubledSmall(n_final=(10, 20), min_sparsity=0.2)
    assert adapt_size(m, n_prev) == n_next


@pytest.mark.parametrize(
    ("m", "n_prev", "n_next"),
    [
        pytest.param(10, (5, 5), (5, 10), id="10-5x5-10x10"),
        pytest.param(10, (4, 5), (4, 10), id="10-4x5-8x10"),
        pytest.param(10, (10, 10), (10, 10), id="10-10x10-10x10"),
        pytest.param(20, (10, 10), (10, 14), id="20-10x10-10x20"),
        pytest.param(20, (10, 20), (10, 20), id="20-10x20-10x20"),
        pytest.param(20, (10, 2), (10, 14), id="20-10x2-10x14"),
    ],
)
def test_scaled_all(m, n_prev, n_next):
    adapt_size = gs.ScaledAll(n_final=(10, 20), min_sparsity=0.2)
    assert adapt_size(m, n_prev) == n_next
