"""Data simulation functionality."""

# Standard
import math
from collections.abc import Callable
from typing import Self

# Third party
import torch
from torch import Tensor, nn


class Signal(nn.Module):
    """
    Basic multidimensional time-domain signal simulator.

    One-dimensional signals are simulated as complex numbers, while
    :math:`d`-dimensional signals are simulated as :math:`2^{d-1}`
    channels of complex numbers according to States convention.
    See :mod:`vrlsnmr.multicomplex` for more details.
    """

    amplitudes: Tensor
    frequencies: Tensor
    decayrates: Tensor
    phases: Tensor

    def __init__(
        self,
        amplitudes: Tensor,
        frequencies: Tensor,
        decayrates: Tensor,
        phases: Tensor,
    ):
        """
        Initialize the simulator.

        Args:
            amplitudes:
                :math:`(k)` or :math:`(b, k)`,
                Amplitude parameters.
            frequencies:
                :math:`(k)`, :math:`(k, d)`, :math:`(b, k)` or :math:`(b, k, d)`,
                Frequency parameters.
            decayrates:
                :math:`(k)`, :math:`(k, d)`, :math:`(b, k)` or :math:`(b, k, d)`,
                Decay rate parameters.
            phases:
                :math:`(k)`, :math:`(k, d)`, :math:`(b, k)` or :math:`(b, k, d)`,
                Phase shift parameters.
        """
        super().__init__()

        assert amplitudes.ndim in (1, 2)
        batched = amplitudes.ndim == 2
        if batched:
            assert frequencies.ndim in (2, 3)
            multidim = frequencies.ndim == 3
        else:
            assert frequencies.ndim in (1, 2)
            multidim = frequencies.ndim == 2

        assert len({frequencies.shape, decayrates.shape, phases.shape}) == 1

        if batched:
            amplitudes = amplitudes.unsqueeze(dim=1)
            frequencies = frequencies.unsqueeze(dim=1)
            decayrates = decayrates.unsqueeze(dim=1)
            phases = phases.unsqueeze(dim=1)

        self.batched = batched
        self.multidim = multidim
        self.register_buffer("amplitudes", amplitudes)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("decayrates", decayrates)
        self.register_buffer("phases", phases)

    def forward(self, t: float | Tensor, *, noise: float = 0.0) -> Tensor:
        """
        Simulate measurement at a given time.

        Args:
            t: :math:`()`, :math:`(m)` or :math:`(m, d)`, Measurement times.
            noise: Noise standard deviation.

        Returns:
            Output values, whose shape depends on the parameter shapes.

            - Unbatched: :math:`(1)`, :math:`(m)` or :math:`(m, c)`.
            - Batched: :math:`(b, 1)`, :math:`(b, m)` or :math:`(b, m, c)`.
        """
        t = 2 * math.pi * t
        if not torch.is_tensor(t):
            t = self.phases.new_full((1,), t)

        if self.multidim:
            out = self._simulate_nd(t)
        else:
            out = self._simulate_1d(t)

        return out + noise * torch.randn_like(out)

    def _simulate_1d(self, t: Tensor) -> Tensor:
        """
        Simulate a one-dimensional signal.

        Args:
            t: :math:`(m)`, Input times.

        Returns:
            :math:`(m)` or :math:`(b, m)`, Output signal values.

        Note:
            When unbatched, all parameters have shape :math:`(k)`, and
            when batched, all parameters have shape :math:`(b, 1, k)`.
        """
        t = t.unsqueeze(dim=-1)  # (m, 1)
        decay = (-self.decayrates * t).exp()  # (m, k) or (b, m, k)
        angle = self.frequencies * t + self.phases
        real = self.amplitudes * decay * angle.cos()
        imag = self.amplitudes * decay * angle.sin()
        out = torch.complex(real, imag).sum(dim=-1)
        return out

    def _simulate_nd(self, t: Tensor) -> Tensor:
        """
        Simulate a multidimensional signal.

        Args:
            t: :math:`(m, d)`, Input times.

        Returns:
            :math:`(m, c)` or :math:`(b, m, c)`, Output signal values,
            where :math:`c = 2^{d-1}`.

        Note:
            When unbatched, amplitudes have shape :math:`(k)` and nonlinear
            parameters have shape :math:`(k, d)`. When batched, amplitudes
            have shape :math:`(b, 1, k)` and nonlinear parameters
            have shape :math:`(b, 1, k, d)`.
        """
        t = t.unsqueeze(dim=-2)  # (m, 1, d)
        decay = (-self.decayrates * t).exp()  # (m, k, d) or (b, m, k, d)
        angle = self.frequencies * t + self.phases
        real = decay * angle.cos()
        imag = decay * angle.sin()

        num_dimensions = imag.size(dim=-1)
        num_channels = 1 << num_dimensions
        dims = torch.arange(num_dimensions, device=t.device)

        out_channels = []
        for channel in range(num_channels):
            mask = (1 << dims) & channel
            phi = torch.where(mask.bool(), imag, real).prod(dim=-1)
            func = (self.amplitudes * phi).sum(dim=-1)
            out_channels.append(func)

        out = torch.stack(out_channels, dim=-1)
        (out_real, out_imag) = out.unflatten(dim=-1, sizes=(-1, 2)).unbind(dim=-1)
        return torch.complex(out_real, out_imag)

    @classmethod
    def build(
        cls,
        *,
        num_components: int = 1,
        num_replicas: int = 1,
        num_dimensions: int = 1,
        frequencies: Callable[[int, int], Tensor],
        decayrates: Callable[[int, int], Tensor],
        amplitudes: Callable[[int, int], Tensor],
        phases: Callable[[int, int], Tensor],
    ) -> Self:
        """
        Build a signal simulator using parameter builders functions.

        Args:
            num_dimensions: :math:`d`, Number of signal dimensions.
            num_components: :math:`k`, Number of signal components (peaks).
            num_replicas: :math:`b`, Number of signal replicas (batch size).
            frequencies: Frequency parameter builder function.
            decayrates: Decay rate parameter builder function.
            amplitudes: Amplitude parameter builder function.
            phases: Phase shift parameter builder function.

        Returns:
            Newly built signal simulator.
        """

        def reshape(params: Tensor) -> Tensor:
            return (
                params.reshape(num_replicas, num_dimensions, num_components)
                .permute(0, 2, 1)
                .squeeze(dim=-1)
                .squeeze(dim=0)
                .contiguous()
            )

        num_copies = num_dimensions * num_replicas
        return cls(
            amplitudes=amplitudes(num_components, num_replicas).squeeze(dim=0),
            frequencies=reshape(frequencies(num_components, num_copies)),
            decayrates=reshape(decayrates(num_components, num_copies)),
            phases=reshape(phases(num_components, num_copies)),
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


def random_normal(n: int, bs: int, *, mean: float, stdev: float) -> Tensor:
    """
    Build a tensor filled with normally distributed values.

    Args:
        n: Number of components.
        bs: Number of replicas.
        mean: Average value.
        stdev: Standard deviation.

    Returns:
        :math:`(bs, n)`, Output tensor.
    """
    return torch.randn(bs, n).mul(stdev).add(mean)


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
    n: int,
    bs: int,
    *,
    lower: float,
    upper: float,
    space: float,
) -> Tensor:
    """
    Build a tensor filled with minimally spaced uniformly distributed values.

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
    out[:, :1] = sample()

    for index in range(1, n):
        trial = out[:, :1].clone()
        while True:
            valid = (trial - out[:, :index]).abs().gt(space).all(dim=1)
            trial = trial.where(valid.unsqueeze(dim=1), sample())
            if valid.all():
                out[:, index] = trial[:, 0]
                break

    return out
