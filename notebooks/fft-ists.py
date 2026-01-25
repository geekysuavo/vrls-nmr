import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import math
    import cmath
    import pickle

    import matplotlib.pyplot as plt
    import torch
    return pickle, plt, torch


@app.cell
def _(pickle):
    with open("instance.pickle", "rb") as file:
        instance = pickle.load(file)
    return (instance,)


@app.cell
def _(instance):
    k, m, n, stdev = (instance.k, instance.m, instance.n, instance.stdev)
    A, x0, noise, y = (instance.A, instance.x0, instance.noise, instance.y)
    B, Phi, ids = (instance.B, instance.Phi, instance.ids)
    return ids, m, n, x0, y


@app.cell
def _():
    mu = 0.98
    return (mu,)


@app.cell
def _(ids, n, torch):
    ids_mask = torch.zeros(n, dtype=torch.bool)
    ids_mask[ids] = True
    return (ids_mask,)


@app.cell
def _(ids_mask, m, mu, n, plt, torch, x0, y):
    xhat = torch.zeros(1, n).cfloat()
    yhat = torch.zeros(1, n).cfloat()
    delta = torch.zeros(1, n).cfloat()

    by = y.view(1, m)

    for it in range(200):
        delta[:, ids_mask] = y - yhat[:, ids_mask]
        xhat += torch.fft.fft(delta, norm="ortho")

        if it == 0:
            thresh = xhat.abs().max(dim=1, keepdim=True).values

        xhat_abs = xhat.abs()
        xhat = torch.where(
            xhat_abs <= thresh,
            0.0,
            xhat * (1 - thresh / xhat_abs),
        )

        yhat = torch.fft.ifft(xhat, norm="ortho")
        thresh *= mu

    plt.plot(xhat[0].real)
    plt.plot(x0.real)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
