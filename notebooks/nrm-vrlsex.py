import marimo

__generated_with = "0.17.2"
app = marimo.App()


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
def _(n, torch):
    mu = torch.zeros(n)
    Gamma = torch.eye(n)
    return Gamma, mu


@app.cell
def _():
    nu_tau = 1
    nu_xi = 1
    return (nu_xi,)


@app.cell
def _(A, Gamma, beta_tau, beta_xi, mu, nu_xi, plt, torch, x0, y):
    for _ in range(100):
        m2 = mu.square() + Gamma.diag()
        nu_w = (nu_xi / (m2 + 1e-09)).sqrt()
        nu_xi_1 = (beta_xi / nu_w.reciprocal().sum()).sqrt()
        ess = (y - A @ mu).square().sum() + (A.t() @ A @ Gamma).diag().sum()
        nu_tau_1 = (beta_tau / ess).sqrt()
        Gamma_1 = torch.linalg.inv(nu_w.diag() + nu_tau_1 * A.t() @ A)
        mu_1 = nu_tau_1 * Gamma_1 @ A.t() @ y
    plt.plot(mu_1)
    plt.plot(x0)
    return nu_tau_1, nu_xi_1


@app.cell
def _(math, torch):
    def inverse_normal(x, nu, lmb):
        return math.sqrt(lmb / (2 * math.pi * nu ** 3)) * torch.exp(-lmb * (_x - nu).square() / (2 * nu ** 2 * _x))
    return (inverse_normal,)


@app.cell
def _(beta_tau, inverse_normal, nu_tau_1, plt, stdev, torch):
    _x = torch.linspace(0, 100000.0, 1000)
    plt.plot(_x, inverse_normal(_x, nu_tau_1.item(), beta_tau))
    plt.scatter([1 / stdev ** 2], [0])
    return


@app.cell
def _(beta_xi, inverse_normal, nu_xi_1, plt, torch):
    _x = torch.linspace(1000.0, 3000.0, 1000)
    plt.plot(_x, inverse_normal(_x, nu_xi_1.item(), beta_xi))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Takeaways

    1. Setting $\beta_\tau = \beta_\xi \gg 1/\sigma^2$ seems to work well. The converged $q(\tau)$ and $q(\xi)$ are
       what you'd expect to set based on knowledge of $\sigma$ and the fixed-$\xi$ experiments.
    2. For experimenting with the "algebraic" changes, the non-extended VRLS can be used.
    3. We can probably relate $\tau / \xi$ to expected SNR.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
