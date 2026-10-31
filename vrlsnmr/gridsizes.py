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
        n_next = list(n_prev)
        ndim = len(self.n_final)
        free_dims = set(range(ndim))
        available_size = m / self.min_sparsity

        while free_dims:
            fixed_dims = set(range(ndim)) - free_dims
            fixed_size = math.prod(n_next[dim] for dim in fixed_dims)
            free_size = math.prod(self.n_final[dim] for dim in free_dims)
            scale = (available_size / fixed_size / free_size) ** (1 / len(free_dims))
            sizes = {dim: 2 * int(scale * self.n_final[dim] / 2) for dim in free_dims}

            # dims cannot shrink below the current grid size.
            dims_at_lower = {dim for dim in sizes if sizes[dim] <= n_prev[dim]}
            if dims_at_lower:
                free_dims -= dims_at_lower
                continue

            # dims cannot grow past the final grid size.
            dims_at_upper = {dim for dim in sizes if sizes[dim] >= self.n_final[dim]}
            for dim in dims_at_upper:
                n_next[dim] = self.n_final[dim]

            free_dims -= dims_at_upper
            if not dims_at_upper:
                for dim in free_dims:
                    n_next[dim] = sizes[dim]
                break

        return tuple(n_next)


@dataclass(frozen=True)
class DoubledAll:
    """Scale all dimensions together to the largest shape by doubling."""

    n_final: tuple[int, ...]
    min_sparsity: float

    def __call__(self, m: int, n_prev: tuple[int, ...]) -> tuple[int, ...]:
        """Adapt the grid shape."""
        n = n_prev
        while any(nd < nf for nd, nf in zip(n, self.n_final)):
            n_next = tuple(min(2 * nd, nf) for nd, nf in zip(n, self.n_final))
            if m / math.prod(n_next) < self.min_sparsity:
                break
            else:
                n = n_next

        return n


@dataclass(frozen=True)
class DoubledSmall:
    """Scale the smallest dimension up to the largest shape by doubling."""

    n_final: tuple[int, ...]
    min_sparsity: float

    def __call__(self, m: int, n_prev: tuple[int, ...]) -> tuple[int, ...]:
        """Adapt the grid shape."""
        n = n_prev
        while any(nd < nf for nd, nf in zip(n, self.n_final)):
            scalable_dims = tuple(
                (nd, dim)
                for dim, (nd, nf) in enumerate(zip(n, self.n_final))
                if nd < nf
            )
            (_, dim) = min(scalable_dims)

            n_list = list(n)
            n_list[dim] = min(2 * n_list[dim], self.n_final[dim])
            n_next = tuple(n_list)
            if m / math.prod(n_next) < self.min_sparsity:
                break
            else:
                n = n_next

        return n
