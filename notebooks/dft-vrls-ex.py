import marimo

__generated_with = "0.17.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
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
    (k, m, n, stdev) = (instance.k, instance.m, instance.n, instance.stdev)
    (A, x0, noise, y) = (instance.A, instance.x0, instance.noise, instance.y)
    Phi = instance.Phi
    return A, Phi, n, x0, y


@app.cell
def _():
    beta_tau = 1e6
    beta_xi = 1e6
    return beta_tau, beta_xi


@app.cell
def _(A, beta_tau, beta_xi, n, plt, torch, x0, y):
    nu_tau = 1
    nu_xi = 1

    mu = torch.zeros(n).cfloat()
    Gamma = torch.eye(n).cfloat()

    for _ in range(100):
        m2 = mu.abs().square() + Gamma.diag().real
        nu_w = (nu_xi / (m2 + 1e-09)).sqrt()
        nu_xi = (beta_xi / nu_w.reciprocal().sum()).sqrt()
        ess = (y - A @ mu).abs().square().sum() + (A.t() @ A @ Gamma).diag().sum().real
        nu_tau = (beta_tau / ess).sqrt()
        Gamma = torch.linalg.inv(nu_w.diag() + nu_tau * A.t().conj() @ A)
        mu = nu_tau * Gamma @ A.t().conj() @ y

    plt.plot(mu.real)
    plt.plot(x0.real)
    return (mu,)


@app.cell
def _(Phi, instance, mu, plt):
    #plt.plot((Phi @ Gamma @ Phi.t().conj()).diag().real);
    plt.plot((Phi @ mu).real)
    plt.plot((Phi @ mu).imag)
    plt.scatter(instance.ids, instance.y.real)
    plt.scatter(instance.ids, instance.y.imag)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Takeaways

    1. Simpler than expected; just make sure weights are real and `.t()` is replaced with `.t().conj()`
    """
    )
    return


if __name__ == "__main__":
    app.run()
