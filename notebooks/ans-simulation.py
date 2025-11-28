import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    import vrlsnmr.algorithms as algo
    import vrlsnmr.simulators as sim
    return Path, algo, partial, plt, sim, torch


@app.cell
def _(partial, sim, torch):
    torch.random.manual_seed(1729)

    ground_truth = sim.Signal.build(
        num_components=8,
        frequencies=partial(sim.random_spaced, lower=-0.5, upper=0.5, space=0.002),
        decayrates=partial(sim.random_chisq, dof=20, mean=0.001),
        amplitudes=partial(sim.random_chisq, dof=50, mean=1.0),
        phases=partial(sim.random_normal, mean=0.0, stdev=1.0e-6),
    )

    ground_truth.cuda()

    dict(ground_truth.named_buffers())
    return (ground_truth,)


@app.cell
def _(algo, ground_truth, partial):
    sigma = 0.1
    tau = 1 / sigma**2
    xi = tau
    niter = 100

    (y, ids, mu, gamma_diag, yhat, sigma_diag) = algo.ans(
        model=algo.vrls,
        measure=partial(ground_truth, noise=sigma),
        m_initial=16,
        m_final=64,
        n_initial=64,
        n_final=2048,
        min_sparsity=0.1,
        tau=tau,
        xi=xi,
        niter=niter,
    )

    y.squeeze_(dim=0)

    mu.squeeze_(dim=0)
    gamma_diag.squeeze_(dim=0)

    yhat = yhat.squeeze(dim=0)
    sigma_diag = sigma_diag.squeeze(dim=0)

    n_fd = mu.size(dim=0)
    n_td = yhat.size(dim=0)
    return ids, mu, n_fd, n_td, niter, sigma, sigma_diag, tau, xi, y, yhat


@app.cell
def _(algo, ground_truth, ids, n_fd, n_td, niter, sigma, tau, torch, xi):
    ids_unif = (
        torch.randperm(n_td, device=ids.device)
        .narrow(dim=0, start=0, length=ids.numel())
    )

    y_unif = ground_truth(ids_unif, noise=sigma).squeeze(dim=0)

    (mu_unif, gamma_unif, _, _) = algo.vrls(
        y_unif,
        ids_unif,
        tau=tau,
        xi=xi,
        n=n_fd,
        niter=niter,
    )

    mu_unif = mu_unif.squeeze(dim=0).roll(n_fd // 2)
    gamma_unif = gamma_unif.squeeze(dim=0).roll(n_fd // 2)
    return (mu_unif,)


@app.cell
def _(ids):
    ids
    return


@app.cell
def _(ground_truth, ids, n_td, plt, sigma_diag, torch, y, yhat):
    t = torch.arange(n_td)

    plt.fill_between(
        t,
        y1=(yhat - sigma_diag.sqrt()).real.cpu(),
        y2=(yhat + sigma_diag.sqrt()).real.cpu(),
        alpha=0.1,
        color="grey",
    )
    plt.plot(t, ground_truth(t.cuda()).real[0].cpu(), alpha=0.2)
    plt.plot(t, yhat.real.cpu(), c="red", alpha=0.2)
    plt.scatter(ids.cpu(), y.real.cpu(), c="orange", s=10)
    return


@app.cell
def _(ground_truth, n_fd, sigma, torch):
    t_fd = torch.arange(n_fd).cuda()
    f = torch.linspace(-0.5, 0.5, n_fd)

    spect = torch.fft.fft(
        ground_truth(t_fd, noise=sigma).squeeze(dim=0),
        norm="ortho",
    ).roll(n_fd // 2)
    return f, spect


@app.cell
def _(y, yhat):
    y.numel(), yhat.numel(), y.numel() / yhat.numel()
    return


@app.cell
def _(Path, f, mu, mu_unif, plt, spect):
    dx = 0.017
    dy = 0.7

    (_, ax) = plt.subplots(figsize=(6, 4))

    ax.plot(
        f + 2 * dx,
        mu_unif.real.cpu() + 2 * dy,
        color=(0.2,) * 3,
    )

    ax.plot(
        f + 1 * dx,
        mu.real.cpu() + 1 * dy,
        color=(0.4,) * 3,
    )

    ax.plot(
        f, spect.real.cpu(),
        color=(0.7,) * 3,
    )

    ax.set_ylabel("Amplitude / a.u.")
    ax.set_xlabel("Normalized frequency / a.u.")
    ax.set_xlim((-0.5, 0.5))
    ax.grid(color=(0.9,) * 3)

    plt.savefig(
        Path.cwd() / "figure-3.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    ax
    return


if __name__ == "__main__":
    app.run()
