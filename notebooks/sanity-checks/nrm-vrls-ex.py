import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import math
    import pickle

    import matplotlib.pyplot as plt
    import torch
    return math, pickle, plt, torch


@app.cell
def _(pickle):
    with open("instance.pickle", "rb") as file:
        instance = pickle.load(file)
    return (instance,)


@app.cell
def _(instance):
    (k, m, n, stdev) = (instance.k, instance.m, instance.n, instance.stdev)
    (A, x0, noise, y) = (instance.A, instance.x0, instance.noise, instance.y)
    return A, n, stdev, x0, y


@app.cell
def _():
    beta_tau = 1e6
    beta_xi = 1e6
    return beta_tau, beta_xi


@app.cell
def _(A, beta_tau, beta_xi, n, plt, torch, x0, y):
    nu_tau = 1
    nu_xi = 1

    mu = torch.zeros(n)
    Gamma = torch.eye(n)

    for _ in range(100):
        m2 = mu.square() + Gamma.diag()
        nu_w = (nu_xi / (m2 + 1e-09)).sqrt()
        nu_xi = (beta_xi / nu_w.reciprocal().sum()).sqrt()
        ess = (y - A @ mu).square().sum() + (A.t() @ A @ Gamma).diag().sum()
        nu_tau = (beta_tau / ess).sqrt()
        Gamma = torch.linalg.inv(nu_w.diag() + nu_tau * A.t() @ A)
        mu = nu_tau * Gamma @ A.t() @ y

    plt.plot(mu)
    plt.plot(x0)
    return nu_tau, nu_xi


@app.cell
def _(math, torch):
    def inverse_normal(x, nu, lmb):
        Z = math.sqrt(lmb / (2 * math.pi * nu ** 3))
        return Z * torch.exp(-lmb * (x - nu).square() / (2 * nu ** 2 * x))
    return (inverse_normal,)


@app.cell
def _(beta_tau, inverse_normal, nu_tau, plt, stdev, torch):
    _x = torch.linspace(0, 100000.0, 1000)
    plt.plot(_x, inverse_normal(_x, nu_tau.item(), beta_tau))
    plt.scatter([1 / stdev ** 2], [0])
    return


@app.cell
def _(beta_xi, inverse_normal, nu_xi, plt, torch):
    _x = torch.linspace(1000.0, 3000.0, 1000)
    plt.plot(_x, inverse_normal(_x, nu_xi.item(), beta_xi))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Takeaways

    1. Setting $\beta_\tau = \beta_\xi \gg 1/\sigma^2$ seems to work well. The converged $q(\tau)$ and $q(\xi)$ are
       what you'd expect to set based on knowledge of $\sigma$ and the fixed-$\xi$ experiments.
    2. For experimenting with the "algebraic" changes, the non-extended VRLS can be used.
    3. We can probably relate $\tau / \xi$ to expected SNR.
    """)
    return


if __name__ == "__main__":
    app.run()
