"""Shared type definitions."""

# Standard
from collections.abc import Callable
from typing import NamedTuple

# Third party
from torch import Tensor


class SamplerReturnType(NamedTuple):
    """
    Result of a single step of active nonuniform sampling.

    Attributes:
        y: :math:`(b, m)`, Time-domain measurements (complex).
        ids: :math:`(m)`, Time-domain measurement indices.
        mu: :math:`(b, n)`, Frequency-domain mean (complex).
        gamma_diag: :math:`(b, n)`, Frequency-domain variance (real, positive).
        yhat: :math:`(b, n)`, Time-domain mean (complex).
        sigma_diag: :math:`(b, n)`, Time-domain variance (real, positive).
    """

    y: Tensor
    ids: Tensor
    mu: Tensor
    gamma_diag: Tensor
    yhat: Tensor
    sigma_diag: Tensor


type PointEstimatorReturnType = tuple[Tensor, Tensor]
type AlgorithmReturnType = tuple[Tensor, Tensor, Tensor, Tensor]
type AlgorithmFunc = Callable[..., AlgorithmReturnType]
type MeasurementFunc = Callable[[int], Tensor]
type CallbackFunc = Callable[[SamplerReturnType], None]
