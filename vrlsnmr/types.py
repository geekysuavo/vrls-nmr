"""Shared type definitions."""

# Standard
from collections.abc import Callable

# Third party
from torch import Tensor

type PointEstimatorReturnType = tuple[Tensor, Tensor]
type AlgorithmReturnType = (
    tuple[Tensor, Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
)
type SamplerReturnType = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
type AlgorithmFunc = Callable[..., AlgorithmReturnType]
type MeasurementFunc = Callable[[Tensor], Tensor]
type CallbackFunc = Callable[[SamplerReturnType], None]

type AdaptSizeIntFunc = Callable[[int, int], int]
type AdaptSizeTupleFunc = Callable[[int, tuple[int, ...]], tuple[int, ...]]
type AdaptSizeFunc = AdaptSizeIntFunc | AdaptSizeTupleFunc
