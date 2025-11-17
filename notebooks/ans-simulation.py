import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import cmath
    from functools import partial
    import math
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn, Tensor
    import torch.nn.functional as F

    from vrlsnmr.algorithms import vrls
    return F, Tensor, math, nn, partial, plt, torch, vrls


@app.cell
def _(torch):
    def constant(n: int, value: float) -> list[float]:
        return torch.full((n,), fill_value=value).tolist()

    def random_chisq(n: int, *, dof: int, mean: float) -> list[float]:
        return torch.randn(n, dof).square().mean(dim=1).mul(mean).tolist()

    def random_unif(n: int, min: float, max: float) -> list[float]:
        return torch.rand(n).mul(max - min).add(min).tolist()

    def random_spaced(n: int, min: float, max: float, space: float) -> list[float]:
        pool = random_unif(n, min, max)
        out = [pool.pop()]
        while len(out) < n:
            if not pool:
                pool = random_unif(n, min, max)

            trial = pool.pop()
            if all(abs(trial) > space for value in out):
                out.append(trial)

        return out
    return constant, random_chisq, random_spaced


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

        @classmethod
        def random(
            cls,
            num_components: int = 1,
            *,
            frequencies: callable,
            decayrates: callable,
            amplitudes: callable,
            phases: callable,
        ) -> "Signal":
            return cls(
                frequencies=frequencies(num_components),
                decayrates=decayrates(num_components),
                amplitudes=amplitudes(num_components),
                phases=phases(num_components),
            )
    return (Signal,)


@app.cell
def _(Signal, constant, partial, random_chisq, random_spaced):
    ground_truth = Signal.random(
        num_components=5,
        frequencies=partial(random_spaced, min=-0.5, max=0.5, space=0.005),
        decayrates=partial(random_chisq, dof=20, mean=0.001),
        amplitudes=partial(random_chisq, dof=3, mean=1.0),
        phases=partial(constant, value=0.0),
    )

    ground_truth.cuda()
    return (ground_truth,)


@app.cell
def _(math):
    def expect_success(m: int, n: int, k: int, margin: float) -> bool:
        return m > margin * k * math.log10(n / k)
    return


@app.function
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
    while n < n_final and m / n >= 0.1:
        n *= 2

    return n


@app.cell
def _(ground_truth, torch, vrls):
    sigma = 0.1
    tau = 1 / sigma**2
    xi = tau
    niter = 100

    m_initial = 16
    m_final = 64

    n_initial = 64
    n_final = 2048

    ids_list = list(range(m_initial))
    ids = torch.tensor(ids_list).cuda()
    y = ground_truth(ids, noise=sigma)

    m = m_initial
    n = n_initial
    iterates = []
    while m < m_final:
        m = len(ids_list)
        n = determine_size(n_initial, n_final, n, m)

        (mu, Gamma_diag, yhat, Sigma_diag) = vrls(y, ids, tau, xi, n, niter)

        mu = mu.roll(n // 2).squeeze(dim=0)
        Gamma_diag = Gamma_diag.roll(n // 2).squeeze(dim=0)

        yhat = yhat.squeeze(dim=0).narrow(dim=0, start=0, length=n // 2)
        Sigma_diag = Sigma_diag.squeeze(dim=0).narrow(dim=0, start=0, length=n // 2)

        iterates.append((mu.cpu(), Gamma_diag.cpu(), yhat.cpu(), Sigma_diag.cpu(), m, n))

        objective = Sigma_diag.clone()
        objective[ids] = 0

        next_index = objective.argmax().item()
        y_next = ground_truth(next_index, noise=sigma)

        ids_list.append(next_index)
        ids = torch.tensor(ids_list).cuda()
        y = torch.cat((y, y_next), dim=0)
    return Sigma_diag, ids, iterates, mu, n, sigma, y, yhat


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
def _(iterates, plt):
    fig, ax = plt.subplots()

    ax.plot([it[-2] for it in iterates], label="m", c="green")
    ax.plot([it[-1] for it in iterates], label="n", c="blue")

    ax2 = ax.twinx()
    ax2.plot([it[-2] / it[-1] for it in iterates], label="m/n", c="red")
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


if __name__ == "__main__":
    app.run()
