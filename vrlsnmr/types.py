"""Shared type definitions."""

# Standard
from collections.abc import Callable

# Third party
from torch import Tensor

type PointEstimatorReturnType = tuple[Tensor, Tensor]
type AlgorithmReturnType = tuple[Tensor, Tensor, Tensor, Tensor]
type SamplerReturnType = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
type AlgorithmFunc = Callable[..., AlgorithmReturnType]
type MeasurementFunc = Callable[[int], Tensor]
type CallbackFunc = Callable[[SamplerReturnType], None]
