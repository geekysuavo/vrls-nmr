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
    return ops, pickle, plt, torch


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
    return Phi, ids, n, stdev, x0, y


@app.cell
def _(stdev, torch):
    tau = torch.full((1,), 1 / stdev**2)  # noise is known
    xi = 1e4
    return tau, xi


@app.cell
def _(ids, n, ops, plt, tau, torch, x0, xi, y):
    mu = torch.zeros(1, n).cfloat()
    w = torch.ones(1, n)

    for _ in range(10):
        K = ops.kernel(w, tau, ids)
        Kinv = torch.linalg.inv(K)
        mu.mul_(0)
        mu[:, ids] = (Kinv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = torch.fft.fft(mu, norm="ortho") / w
        Gamma_diag = ops.xmarginal(Kinv, w, ids)
        m2 = mu.abs().square() + Gamma_diag
        w = (xi / (m2 + 1e-09)).sqrt()

    plt.plot(mu[0].real)
    plt.plot(x0.real)
    return Kinv, mu, w


@app.cell
def _(Kinv, Phi, ids, instance, mu, n, ops, plt, torch, w):
    #plt.plot((Phi @ Gamma @ Phi.t().conj()).diag().real);
    t = torch.arange(n)
    yhat = (Phi @ mu.unsqueeze(dim=-1)).squeeze(dim=-1).imag[0]
    s = ops.ymarginal(Kinv, w, ids)[0]
    width = 1000.0
    plt.fill_between(t, y1=yhat + width * s, y2=yhat - width * s, alpha=0.2)
    plt.plot(yhat)
    plt.scatter(instance.ids, instance.y.imag)
    return


if __name__ == "__main__":
    app.run()
