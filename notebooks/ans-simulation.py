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
                t = torch.tensor(t, dtype=self.amplitudes.dtype, device=self.amplitudes.device)

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

    ground_truth.cuda()
    return (ground_truth,)


@app.cell
def _(math):
    def expect_success(m: int, n: int, k: int, margin: float) -> bool:
        return m > margin * k * math.log10(n / k)
    return (expect_success,)


@app.cell
def _(expect_success):
    def determine_size(
        n_initial: int,
        n_final: int,
        n_prev: int,
        m: int,
        eps: float = 0.2,
        margin: float = 1.5,
    ) -> int:
        k_prev = int(eps * n_prev)
        n = n_prev
        while n < n_final and expect_success(m, 2 * n, k_prev, margin):
            n *= 2

        return n
    return (determine_size,)


@app.cell
def _(determine_size, ground_truth, torch, vrls):
    sigma = 0.1
    tau = 1 / sigma**2
    xi = tau
    m_initial = 16
    n_initial = 64
    n_final = 2048
    niter = 100
    nsamp = 10

    ids_list = list(range(m_initial))
    ids = torch.tensor(ids_list).cuda()
    y = ground_truth(ids, noise=sigma)

    n = n_initial
    iterates = []
    for _ in range(nsamp):
        m = len(ids_list)
        n = determine_size(n_initial, n_final, n, m)

        (mu, Gamma_diag, yhat, Sigma_diag) = vrls(y, ids, tau, xi, n, niter)

        mu = mu.roll(n // 2).squeeze(dim=0)
        Gamma_diag = Gamma_diag.roll(n // 2).squeeze(dim=0)

        yhat = yhat.squeeze(dim=0).narrow(dim=0, start=0, length=n // 2)
        Sigma_diag = Sigma_diag.squeeze(dim=0).narrow(dim=0, start=0, length=n // 2)

        iterates.append((mu.cpu(), Gamma_diag.cpu(), yhat.cpu(), Sigma_diag.cpu()))

        objective = Sigma_diag.clone()
        objective[ids] = 0

        next_index = objective.argmax().item()
        y_next = ground_truth(next_index, noise=sigma)

        ids_list.append(next_index)
        ids = torch.tensor(ids_list).cuda()
        y = torch.cat((y, y_next), dim=0)
    return Gamma_diag, Sigma_diag, ids, iterates, mu, n, sigma, y, yhat


@app.cell
def _(ids):
    ids
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
    plt.plot(t, ground_truth(t.cuda()).real.cpu(), alpha=0.2)
    plt.plot(t, yhat.real.cpu(), c="red", alpha=0.2)
    plt.scatter(ids.cpu(), y.real.cpu(), c="orange", s=10)
    return (t,)


@app.cell
def _(iterates, plt):
    plt.plot(iterates[0][3])
    plt.plot(iterates[1][3])
    #plt.plot(iterates[10][3])
    #plt.plot(iterates[20][3])
    plt.plot(iterates[-1][3])
    return


@app.cell
def _(F, ground_truth, mu, n, plt, sigma, t, torch):
    spect = torch.fft.fft(
        F.pad(ground_truth(t.cuda(), noise=sigma), (0, n // 2)),
        norm="ortho",
    ).roll(n // 2)

    plt.plot(mu.real.cpu())
    plt.plot(spect.real.cpu(), alpha=0.5)
    return


@app.cell
def _(mu):
    K = mu.real.gt(0.5).sum()
    K
    return


@app.cell
def _(Gamma_diag, mu):
    (mu.abs() > Gamma_diag.sqrt()).sum()
    return


@app.cell
def _(Gamma_diag, mu, plt):
    plt.plot(mu.abs().cpu())
    plt.ylim((-0.1, 1))
    plt.plot(Gamma_diag.sqrt().cpu())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
