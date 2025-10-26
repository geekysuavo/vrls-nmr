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
    return A, m, n, stdev, x0, y


@app.cell
def _(stdev):
    tau = 1 / stdev**2  # noise is known
    xi = 1e4
    return tau, xi


@app.cell
def _(n, torch):
    _mu = torch.zeros(n)
    Gamma = torch.eye(n)
    w = torch.ones(n)
    return (w,)


@app.cell
def _(A, m, plt, tau, torch, w, x0, xi, y):
    for _ in range(100):
        Winv = w.reciprocal().diag()
        K = torch.eye(m) / tau + A @ Winv @ A.t()
        Kinv = torch.linalg.inv(K)
        _mu = A.t() @ Kinv @ y / w
        Gamma_diag = w.reciprocal() - torch.sum(Kinv @ A / w * A / w, dim=0)
        m2 = _mu.square() + Gamma_diag
        w_1 = (xi / (m2 + 1e-09)).sqrt()
    plt.plot(_mu)
    plt.plot(x0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Takeaways

    1. The Woodbury identity is super handy after all.
    2. The next step is a partial Fourier VRLS notebook.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
