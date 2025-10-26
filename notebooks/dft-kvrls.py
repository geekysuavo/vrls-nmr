import marimo

__generated_with = "0.17.2"
app = marimo.App()


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
    return A, Phi, m, n, stdev, x0, y


@app.cell
def _(stdev):
    tau = 1 / stdev**2  # noise is known
    xi = 1e4
    return tau, xi


@app.cell
def _(n, torch):
    mu = torch.zeros(n).cfloat()
    Gamma = torch.eye(n).cfloat()
    w = torch.ones(n)
    return (w,)


@app.cell
def _(A, m, plt, tau, torch, w, x0, xi, y):
    for _ in range(100):
        Winv = w.reciprocal().diag().cfloat()
        K = torch.eye(m) / tau + A @ Winv @ A.t().conj()
        Kinv = torch.linalg.inv(K)
        mu_1 = A.t().conj() @ Kinv @ y / w
        Gamma_diag = w.reciprocal() - torch.sum(Kinv @ A / w * A.conj() / w, dim=0)
        m2 = mu_1.abs().square() + Gamma_diag.real
        w_1 = (xi / (m2 + 1e-09)).sqrt()
    plt.plot(mu_1.real)
    plt.plot(x0.real)
    return (mu_1,)


@app.cell
def _(Phi, instance, mu_1, plt):
    #plt.plot((Phi @ Gamma @ Phi.t().conj()).diag().real);
    plt.plot((Phi @ mu_1).real)
    plt.plot((Phi @ mu_1).imag)
    plt.scatter(instance.ids, instance.y.real)
    plt.scatter(instance.ids, instance.y.imag)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Takeaways

    1. Easy peasy, lemon squeezy.
    2. Double-check this using an actual NUS NMR FID.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
