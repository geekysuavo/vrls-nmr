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
    return Phi, ids, m, n, x0, y


@app.cell
def _():
    beta_tau = 1e6
    beta_xi = 1e6
    return beta_tau, beta_xi


@app.cell
def _(beta_tau, beta_xi, ids, m, n, ops, plt, torch, x0, y):
    mu = torch.zeros(1, n).cfloat()
    nu_w = torch.ones(1, n)
    nu_tau = torch.ones(1)
    nu_xi = torch.ones(1)

    by = y.view(1, m)

    for _ in range(10):
        K = ops.kernel(nu_w, nu_tau, ids)
        Kinv = torch.linalg.inv(K)
        mu.mul_(0)
        mu[:, ids] = (Kinv @ y.unsqueeze(dim=-1)).squeeze(dim=-1)
        mu = torch.fft.fft(mu, norm="ortho") / nu_w
        Gamma_diag = ops.xmarginal(Kinv, nu_w, ids)
        Sigma_diag = ops.ymarginal(Kinv, nu_w, ids)
        m2 = mu.abs().square() + Gamma_diag
        nu_w = (nu_xi / (m2 + 1e-09)).sqrt()
        nu_xi = (beta_xi / nu_w.reciprocal().sum(dim=-1)).sqrt()
        yhat = torch.fft.ifft(mu, norm="ortho")[:, ids]
        err = (by - yhat).abs().square().sum(dim=-1)
        ess = err + Sigma_diag[:, ids].sum(dim=-1) / n
        nu_tau = (beta_tau / ess).sqrt()

    plt.plot(mu[0].real)
    plt.plot(x0.real)
    return Kinv, mu, nu_w


@app.cell
def _(Kinv, Phi, ids, instance, mu, n, nu_w, ops, plt, torch):
    #plt.plot((Phi @ Gamma @ Phi.t().conj()).diag().real);
    t = torch.arange(n)
    yhat_real = (Phi @ mu.unsqueeze(dim=-1)).squeeze(dim=-1).real[0]
    s = ops.ymarginal(Kinv, nu_w, ids)[0]
    width = 1000.0
    plt.fill_between(
        t,
        y1=yhat_real + width * s,
        y2=yhat_real - width * s,
        alpha=0.2,
    )
    plt.plot(yhat_real)
    plt.scatter(instance.ids, instance.y.real)
    return


if __name__ == "__main__":
    app.run()
