import torch
from torch import Tensor

import vrlsnmr.operators as op


@torch.inference_mode()
def vrls(
    y: Tensor,
    ids: Tensor,
    tau: float,
    xi: float,
    n: int,
    niter: int,
    eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor]:
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    w = torch.ones(n, dtype=real_dtype, device=device)
    mu = torch.zeros(n, dtype=complex_dtype, device=device)
    tau = torch.tensor(tau, dtype=real_dtype, device=device)

    for _ in range(niter):
        K = op.kernel(w, ids, tau)
        Kinv = torch.linalg.inv(K)

        mu.mul_(0)
        mu[ids] = Kinv @ y
        mu = torch.fft.fft(mu, norm="ortho") / w

        Gamma_diag = op.xmarginal(Kinv, w, ids)

        m2 = mu.abs().square() + Gamma_diag
        w = (xi / (m2 + eps)).sqrt()

    return (mu, Gamma_diag)
