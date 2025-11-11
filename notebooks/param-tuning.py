import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import cmath
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn, Tensor
    import torch.nn.functional as F

    from vrlsnmr.algorithms import vrls
    return F, Tensor, math, nn, plt, torch, vrls


@app.cell
def _(Tensor, math, nn, torch):
    class Signal(nn.Module):
        def __init__(
            self,
            frequencies: list[float],
            decayrates: list[float],
            amplitudes: list[float],
            phases: list[float],
        ):
            super().__init__()

            frequencies = torch.tensor(frequencies)
            decayrates = torch.tensor(decayrates)
            amplitudes = torch.tensor(amplitudes)
            phases = torch.tensor(phases)

            self.register_buffer("frequencies", frequencies)
            self.register_buffer("decayrates", decayrates)
            self.register_buffer("amplitudes", amplitudes)
            self.register_buffer("phases", phases)

        def forward(self, t: int | float | Tensor, *, noise: float = 0.0) -> Tensor:
            t = 2 * math.pi * t
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=self.amplitudes.dtype)

            while t.ndim < 2:
                t = t.unsqueeze(dim=-1)

            real = (self.amplitudes * (self.frequencies * t).cos() * (-self.decayrates * t).exp())
            imag = (self.amplitudes * (self.frequencies * t).sin() * (-self.decayrates * t).exp())
            ph = torch.complex(self.phases.cos(), self.phases.sin())
            out = (ph * torch.complex(real, imag)).sum(dim=1)
            eps = noise * torch.randn_like(out)
            return out + eps
    return (Signal,)


@app.cell
def _(Signal):
    ground_truth = Signal(
        frequencies=[-0.083, -0.026, 0.038, 0.042, 0.074, 0.091],
        decayrates=[0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        amplitudes=[1.0, 1.0, 0.1, 0.3, 1.0, 0.9],
        phases=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return (ground_truth,)


@app.cell
def _():
    sched = [
        1, 2, 3, 5, 6, 8, 9, 11, 13, 16,
        17, 19, 21, 25, 29, 33, 38, 46, 56, 58,
        69, 73, 82, 98, 116, 118, 138, 165, 187, 199,
        217, 235, 267, 310, 365, 408, 415, 457, 475, 510,
        595, 615, 648, 660, 667, 738, 851, 872, 873, 959,
        991,
    ]
    return (sched,)


@app.cell
def _(ground_truth, sched, torch):
    sigma = 0.1
    tau = 1 / sigma**2
    xi = tau
    n = 2048
    niter = 20

    ids = torch.tensor(sched) - 1
    y = ground_truth(ids, noise=sigma)

    ids = ids.cuda()
    y = y.cuda()
    return ids, n, niter, sigma, tau, xi, y


@app.cell
def _(ids, n, niter, tau, vrls, xi, y):
    (mu, Gamma_diag, yhat, Sigma_diag) = vrls(y, ids, tau, xi, n, niter)

    mu = mu.roll(n // 2).squeeze(dim=0)
    Gamma_diag = Gamma_diag.squeeze(dim=0)

    yhat = yhat.squeeze(dim=0).narrow(dim=0, start=0, length=n // 2)
    Sigma_diag = Sigma_diag.squeeze(dim=0).narrow(dim=0, start=0, length=n // 2)
    return Sigma_diag, mu, yhat


@app.cell
def _():
    return


@app.cell
def _(Sigma_diag, ground_truth, ids, n, plt, torch, y, yhat):
    t = torch.arange(n // 2)

    plt.fill_between(
        t,
        y1=(yhat - Sigma_diag.sqrt()).real.cpu(),
        y2=(yhat + Sigma_diag.sqrt()).real.cpu(),
        alpha=0.1,
        color="grey",
    )
    plt.plot(t, ground_truth(t).real, alpha=0.2)
    plt.plot(t, yhat.real.cpu(), c="red", alpha=0.2)
    plt.scatter(ids.cpu(), y.real.cpu(), c="orange", s=10)
    return (t,)


@app.cell
def _(F, ground_truth, mu, n, plt, sigma, t, torch):
    spect = torch.fft.fft(
        F.pad(ground_truth(t, noise=sigma), (0, n // 2)),
        norm="ortho",
    ).roll(n // 2)

    plt.plot(mu.real.cpu())
    plt.plot(spect.real.cpu(), alpha=0.5)
    return


if __name__ == "__main__":
    app.run()
