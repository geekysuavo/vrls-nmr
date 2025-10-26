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
    return A, Phi, n, stdev, x0, y


@app.cell
def _(stdev):
    tau = 1 / stdev**2  # noise is known
    xi = 1e4
    return tau, xi


@app.cell
def _(A, n, plt, tau, torch, x0, xi, y):
    mu = torch.zeros(n).cfloat()
    Gamma = torch.eye(n).cfloat()

    for _ in range(100):
        m2 = mu.abs().square() + Gamma.diag().real
        w = (xi / (m2 + 1e-09)).sqrt()
        Gamma = torch.linalg.inv(w.diag() + tau * A.t().conj() @ A)
        mu = tau * Gamma @ A.t().conj() @ y

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
