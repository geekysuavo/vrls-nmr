import math
from typing import Callable

import torch
from torch import nn, Tensor


class Signal(nn.Module):
    """
    Basic one-dimensional signal (FID) simulator.

    Args:
        frequencies: :math:`(k)` or `(b, k)`, Frequencies.
        decayrates: :math:`(k)` or :math:`(b, k)`, Decay rates.
        amplitudes: :math:`(k)` or :math:`(b, k)`, Amplitudes.
        phases: :math:`(k)` or :math:`(b, k)`, Phase shifts.
    """

    def __init__(
        self,
        frequencies: Tensor,
        decayrates: Tensor,
        amplitudes: Tensor,
        phases: Tensor,
    ):
        super().__init__()

        shapes = {
            frequencies.shape,
            decayrates.shape,
            amplitudes.shape,
            phases.shape,
        }
        assert len(shapes) == 1

        shape = shapes.pop()
        ndim = len(shape)
        assert ndim in (1, 2)

        self._batched = ndim == 2
        if self._batched:
            frequencies = frequencies.unqsueeze(dim=1)
            decayrates = decayrates.unqsueeze(dim=1)
            amplitudes = amplitudes.unqsueeze(dim=1)
            phases = phases.unqsueeze(dim=1)

        self.register_buffer("frequencies", frequencies)
        self.register_buffer("decayrates", decayrates)
        self.register_buffer("amplitudes", amplitudes)
        self.register_buffer("phases", phases)

    def forward(self, t: int | float | Tensor, *, noise: float = 0.0) -> Tensor:
        """
        Simulate measurement at a given time.

        Args:
            t: :math:`()` or :math:`(n)`, Measurement times.

        Returns:
            Output values, whose shape depends on the parameter shapes.

            - Unbatched: :math:`(1)` or :math:`(n)`.
            - Batched: :math:`(b, 1)` or :math:`(b, n)`.
        """
        t = 2 * math.pi * t
        if not torch.is_tensor(t):
            t = self.phases.new_full((1,), t)

        if self._batched:
            t = t.unsqueeze(dim=0)

        t = t.unsqsueeze(dim=-1)

        decay = torch.exp(-self.decayrates * t)
        real = torch.cos(self.frequencies * t)
        imag = torch.sin(self.frequencies * t)
        ph = torch.complex(self.phases.cos(), self.phases.sin())
        out = self.amplitudes * decay * ph * torch.complex(real, imag)
        out = out.sum(dim=-1)
        return out + torch.randn_like(out)

    @classmethod
    def build(
        cls,
        num_components: int = 1,
        num_replicas: int = 1,
        *,
        frequencies: Callable[[int, int], Tensor],
        decayrates: Callable[[int, int], Tensor],
        amplitudes: Callable[[int, int], Tensor],
        phases: Callable[[int, int], Tensor],
    ) -> "Signal":
        """
        Build a signal simulator using parameter builders functions.

        Args:
            num_components: :math:`k`, Number of signal components.
            num_replicas: :math:`b`, Number of signal replicas (batch size).
            frequencies: Frequency parameter builder function.
            decayrates: Decay rate parameter builder function.
            amplitudes: Amplitude parameter builder function.
            phases: Phase shift parameter builder function.

        Returns:
            Newly built signal simulator.
        """
        return cls(
            frequencies=frequencies(num_components, num_replicas),
            decayrates=decayrates(num_components, num_replicas),
            amplitudes=amplitudes(num_components, num_replicas),
            phases=phases(num_components, num_replicas),
        )


def constant(n: int, bs: int, *, value: float) -> Tensor:
    """
    Build a tensor filled with constant values.

    Args:
        n: Number of components.
        bs: Number of replicas.
        value: Constant value.

    Returns:
        :math:`(bs, n)`, Output tensor.
    """
    return torch.full((bs, n), fill_value=value)


def random_chisq(n: int, bs: int, *, dof: int, mean: float) -> Tensor:
    """
    Build a tensor filled with scaled chi-squared distributed values.

    Args:
        n: Number of components.
        bs: Number of replicas.
        dof: Degrees of freedom.
        mean: Scale factor.

    Returns:
        :math:`(bs, n)`, Output tensor.
    """
    return torch.randn(bs, n, dof).square().mean(dim=2).mul(mean)


def random_unif(n: int, bs: int, *, lower: float, upper: float) -> Tensor:
    """
    Build a tensor filled with uniformly distributed values.

    Args:
        n: Number of components.
        bs: Number of replicas.
        lower: Minimum value.
        upper: Maximum value.

    Returns:
        :math:`(bs, n)`, Output tensor.
    """
    return torch.rand(bs, n).mul(upper - lower).add(lower)


def random_spaced(
    n: int, bs: int, *, lower: float, upper: float, space: float
) -> Tensor:
    """
    Build a tensor filled with uniformly distributed values such that
    no two values are within a minimum spacing.

    Args:
        n: Number of components.
        bs: Number of replicas.
        lower: Minimum value.
        upper: Maximum value.
        space: Minimum spacing.

    Returns:
        :math:`(bs, n)`, Output tensor.
    """
    def sample():
        return random_unif(1, bs, lower=lower, upper=upper)

    out = torch.full((bs, n), fill_value=torch.nan)
    out[:, 0] = sample()

    for index in range(1, n):
        trial = out[:, :1].clone()
        while True:
            valid = (trial - out[:, :index]).abs().gt(space).all(dim=1)
            trial = trial.where(valid.unsqueeze(dim=1), sample())
            if valid.all():
                out[:, index] = trial[:, 0]
                break

    return out
