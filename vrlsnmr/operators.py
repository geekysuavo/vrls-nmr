from pathlib import Path

import torch
from torch import Tensor

torch.ops.load_library(Path(__file__).parent / "_ops.so")


def kernel(weights: Tensor, tau: Tensor, ids: Tensor) -> Tensor:
    return torch.ops.vrlsnmr.kernel(weights, tau, ids)


def xmarginal(kernel_inv: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
    return torch.ops.vrlsnmr.xmarginal(kernel_inv, weights, ids)


def ymarginal(kernel_inv: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
    return torch.ops.vrlsnmr.ymarginal(kernel_inv, weights, ids)
