import marimo

__generated_with = "0.25.1"
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
    from vrlsnmr.algorithms import ists

    return F, ists, math, md, mx, np, partial, plt, sim, torch


@app.cell
def _(md, partial, sim, torch):
    measure = sim.Signal.build(
        num_components=10,
        num_dimensions=2,
        amplitudes=partial(sim.random_chisq, dof=20, mean=1.0),
        frequencies=partial(sim.random_unif, lower=0.05, upper=0.45),
        decayrates=partial(sim.random_chisq, dof=10, mean=0.001),
        phases=partial(sim.constant, value=0),
    )

    ids = md.unravel_ids(
        torch.randperm(128 * 128).narrow(dim=0, start=0, length=800),
        shape=(128, 128),
    )

    #ids = op.schedexp(rate=0.01, m=50, n=1000)
    y = measure(ids, noise=0.1)
    return ids, measure, y


@app.cell
def _(ids, ists, y):
    mu = 0.98
    n = (256, 256)
    (xhat, yhat) = ists(y, ids, mu, n, niter=500)
    return n, xhat


@app.cell
def _(F, measure, mx, torch):
    t = torch.cartesian_prod(
        torch.arange(128),
        torch.arange(128),
    )
    y0 = F.pad(
        measure(t).unsqueeze(dim=0).view(1, 2, 128, 128),
        (0, 128, 0, 128),
    )
    y0_ft = mx.fftn(y0, states=True)
    return (y0_ft,)


@app.cell
def _(np, plt, y0_ft):
    plt.contour(
        y0_ft[0, 0].real[:128, :128],
        levels=np.linspace(1, 50, 10),
    )
    return


@app.cell
def _(np, plt, xhat):
    plt.contour(
        xhat[0, 0].real[:128, :128],
        levels=np.linspace(1, 50, 10),
    )
    return


@app.cell
def _(plt, xhat):
    plt.imshow(
        xhat[0, 0].real[:128, :128],
        origin="lower",
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
    )
    return


@app.cell
def _(plt, xhat):
    plt.plot(xhat[0, 0].real.sum(dim=0)[:128])
    return


@app.cell
def _(plt, xhat):
    plt.plot(xhat[0, 0].real.sum(dim=1)[:128])
    return


@app.cell
def _(ids, plt):
    plt.scatter(*ids.unbind(dim=1))
    return


@app.cell
def _(ids, math, n):
    2**len(n) * ids.shape[0] / math.prod(n)
    return


if __name__ == "__main__":
    app.run()
