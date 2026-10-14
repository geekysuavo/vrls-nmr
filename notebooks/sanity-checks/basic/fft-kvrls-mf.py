import marimo

__generated_with = "0.17.2"
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

    import vrlsnmr.operators as ops
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
    return ids, m, n, stdev, x0, y


@app.cell
def _(stdev):
    tau = 1 / stdev**2  # noise is known
    xi = 1e4

    L = 1.0
    return L, tau, xi


@app.cell
def _(ids, n, torch):
    ids_mask = torch.zeros(n, dtype=torch.bool)
    ids_mask[ids] = True
    return (ids_mask,)


@app.cell
def _(L, ids_mask, m, n, plt, tau, torch, x0, xi, y):
    mu = torch.zeros(1, n).cfloat()
    gamma = torch.ones(1, n)

    by = y.view(1, m)

    for _ in range(100):
        m2 = mu.abs().square() + gamma
        w = (xi / (m2 + 1e-09)).sqrt()

        delta = torch.fft.ifft(mu, norm="ortho")
        delta[:, ~ids_mask] = 0
        delta[:, ids_mask] -= y
        delta = torch.fft.fft(delta, norm="ortho")
        delta = (L / 2) * mu - delta

        D_diag = (tau / 2) + w
        mu = tau * D_diag.reciprocal() * delta

        gamma = (tau * (m / n) + w).reciprocal()

    plt.plot(mu[0].real)
    plt.plot(x0.real)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
