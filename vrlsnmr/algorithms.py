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
    """
    Basic VRLS.

    Args:
        y: :math:`(b, m)`, Measurements (complex).
        ids: :math:`(m)`, Measurement indices.
        tau: Fixed noise precision.
        xi: Fixed weight scale.
        n: Sparse dimension size.
        niter: Number of iterations.
        eps: Weight update positivity parameter.

    Returns:
        A :class:`tuple` containing

        - :math:`(b, n)`, Sparse mean (complex).
        - :math:`(b, n)`, Sparse variance (real, positive).
    """
    device = y.device
    complex_dtype = y.dtype
    real_dtype = y.dtype.to_real()

    if y.ndim == 1:
        y = y.unsqueeze(dim=0)

    bs = y.size(dim=0)

    w = torch.ones((bs, n), dtype=real_dtype, device=device)
    mu = torch.zeros((bs, n), dtype=complex_dtype, device=device)
    tau = torch.full((bs,), tau, dtype=real_dtype, device=device)

    for _ in range(niter):
        K = op.kernel(w, tau, ids)
        Kinv = torch.linalg.inv(K)

        mu.mul_(0)
        mu[:, ids] = (Kinv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = torch.fft.fft(mu, norm="ortho") / w

        Gamma_diag = op.xmarginal(Kinv, w, ids)

        m2 = mu.abs().square() + Gamma_diag
        w = (xi / (m2 + eps)).sqrt()

    return (mu, Gamma_diag)
