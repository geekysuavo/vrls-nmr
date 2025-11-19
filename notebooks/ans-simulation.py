import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial

    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    import vrlsnmr.algorithms as algo
    import vrlsnmr.simulators as sim
    return algo, partial, plt, sim, torch


@app.cell
def _(partial, sim):
    ground_truth = sim.Signal.build(
        num_components=8,
        frequencies=partial(sim.random_spaced, lower=-0.5, upper=0.5, space=0.002),
        decayrates=partial(sim.random_chisq, dof=20, mean=0.001),
        amplitudes=partial(sim.random_chisq, dof=10, mean=1.0),
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
    return ids, mu, sigma, sigma_diag, y, yhat


@app.cell
def _(ids):
    ids
    return


@app.cell
def _(ground_truth, ids, plt, sigma_diag, torch, y, yhat):
    n_td = yhat.size(dim=1)
    t = torch.arange(n_td)

    plt.fill_between(
        t,
        y1=(yhat[0] - sigma_diag[0].sqrt()).real.cpu(),
        y2=(yhat[0] + sigma_diag[0].sqrt()).real.cpu(),
        alpha=0.1,
        color="grey",
    )
    plt.plot(t, ground_truth(t.cuda()).real[0].cpu(), alpha=0.2)
    plt.plot(t, yhat[0].real.cpu(), c="red", alpha=0.2)
    plt.scatter(ids.cpu(), y.real[0].cpu(), c="orange", s=10)
    return


@app.cell
def _(ground_truth, mu, plt, sigma, torch):
    n_fd = mu.size(dim=1)
    t_fd = torch.arange(n_fd).cuda()

    spect = torch.fft.fft(
        ground_truth(t_fd, noise=sigma)[0],
        norm="ortho",
    ).roll(n_fd // 2)

    plt.plot(mu[0].real.cpu())
    plt.plot(spect.real.cpu(), alpha=0.5)
    return


if __name__ == "__main__":
    app.run()
