import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import math
    from functools import partial

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    import vrlsnmr.multicomplex as mx
    import vrlsnmr.multidim as md
    import vrlsnmr.operators as op
    import vrlsnmr.simulators as sim
    from vrlsnmr.algorithms import vrls_ex

    return F, mx, partial, plt, sim, torch, vrls_ex


@app.cell
def _(partial, sim, torch):
    measure = sim.Signal.build(
        num_components=10,
        amplitudes=partial(sim.random_chisq, dof=20, mean=1.0),
        frequencies=partial(sim.random_unif, lower=0.05, upper=0.45),
        decayrates=partial(sim.random_chisq, dof=10, mean=0.001),
        phases=partial(sim.constant, value=0),
    )

    ids = torch.randperm(512).narrow(dim=0, start=0, length=128)
    y = measure(ids, noise=0.1)
    return ids, measure, y


@app.cell
def _(ids, vrls_ex, y):
    beta_xi = beta_tau = 1e6
    n = 1024
    (mu, gamma, yhat, sigma) = vrls_ex(y, ids, beta_tau, beta_xi, n, niter=50)
    return mu, n


@app.cell
def _(F, measure, mx, torch):
    t = torch.arange(512)
    y0 = F.pad(
        measure(t).unsqueeze(dim=0),
        (0, 512),
    )
    y0_ft = mx.fftn(y0, states=True)
    return (y0_ft,)


@app.cell
def _(plt, y0_ft):
    plt.plot(y0_ft[0].real[:512])
    return


@app.cell
def _(mu, plt):
    plt.plot(mu[0].real[:512])
    return


@app.cell
def _(ids, n):
    2 * ids.shape[0] / n
    return


if __name__ == "__main__":
    app.run()
