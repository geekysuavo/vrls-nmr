"""Grid size adaptation methods."""

# Standard
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Scaled:
    """Scale to the largest even size."""

    n_final: int
    min_sparsity: float

    def __call__(self, m: int, n_prev: int) -> int:
        """Adapt the grid size."""
        scale = m / (self.min_sparsity * self.n_final)
        n_next = 2 * int(scale * self.n_final / 2)
        return max(min(n_next, self.n_final), n_prev)


@dataclass(frozen=True)
class Doubled:
    """Scale to the largest size by doubling."""

    n_final: int
    min_sparsity: float

    def __call__(self, m: int, n_prev: int) -> int:
        """Adapt the grid size."""
        n = n_prev
        while n < self.n_final:
            n_next = min(2 * n, self.n_final)
            if m / n_next < self.min_sparsity:
                break
            else:
                n = n_next

        return n


@dataclass(frozen=True)
class ScaledAll:
    """Scale all dimensions together to the largest even shape."""

    n_final: tuple[int, ...]
    min_sparsity: float

    def __call__(self, m: int, n_prev: tuple[int, ...]) -> tuple[int, ...]:
        """Adapt the grid shape."""
        ndim = len(self.n_final)
        n_total = math.prod(self.n_final)
        scale = (m / (self.min_sparsity * n_total)) ** (1 / ndim)
        n_next = tuple(2 * int(scale * nf / 2) for nf in self.n_final)
        return tuple(
            max(min(nn, nf), nd) for nn, nd, nf in zip(n_next, n_prev, self.n_final)
        )


@dataclass(frozen=True)
class DoubledAll:
    """Scale all dimensions together to the largest shape by doubling."""

    n_final: tuple[int, ...]
    min_sparsity: float

    def __call__(self, m: int, n_prev: tuple[int, ...]) -> tuple[int, ...]:
        """Adapt the grid shape."""
        n = n_prev
        while (
            any(nd < nf for nd, nf in zip(n, self.n_final))
            and m / math.prod(n) >= self.min_sparsity
        ):
            n = tuple(min(2 * nd, nf) for nd, nf in zip(n, self.n_final))

        return n


@dataclass(frozen=True)
class DoubledSmall:
    """Scale the smallest dimension up to the largest shape by doubling."""

    n_final: tuple[int, ...]
    min_sparsity: float

    def __call__(self, m: int, n_prev: tuple[int, ...]) -> tuple[int, ...]:
        """Adapt the grid shape."""
        n = n_prev
        while (
            any(nd < nf for nd, nf in zip(n, self.n_final))
            and m / math.prod(n) >= self.min_sparsity
        ):
            scalable_dims = tuple(
                (nd, dim)
                for dim, (nd, nf) in enumerate(zip(n, self.n_final))
                if nd < nf
            )
            (_, dim) = min(scalable_dims)

            n_list = list(n)
            n_list[dim] = min(2 * n_list[dim], self.n_final[dim])
            n = tuple(n_list)

        return n
