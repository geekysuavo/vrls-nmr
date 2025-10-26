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
    return A, n, stdev, x0, y


@app.cell
def _(stdev):
    tau = 1 / stdev**2  # noise is known
    xi = 1e4
    return tau, xi


@app.cell
def _(n, torch):
    mu = torch.zeros(n)
    Gamma = torch.eye(n)
    return Gamma, mu


@app.cell
def _(Gamma, mu, xi):
    _m2 = mu.square() + Gamma.diag()
    w = (xi / (_m2 + 1e-09)).sqrt()
    return (w,)


@app.cell
def _(A, tau, torch, w):
    Gamma_1 = torch.linalg.inv(w.diag() + tau * A.t() @ A)
    return (Gamma_1,)


@app.cell
def _(A, Gamma_1, tau, y):
    mu_1 = tau * Gamma_1 @ A.t() @ y
    return (mu_1,)


@app.cell
def _(Gamma_1, mu_1, w):
    w_init = w.clone()
    Gamma_init = Gamma_1.clone()
    mu_init = mu_1.clone()
    return


@app.cell
def _(mu_1, plt, x0):
    plt.plot(mu_1)
    plt.plot(x0)
    return


@app.cell
def _(A, Gamma_1, mu_1, plt, tau, torch, x0, xi, y):
    for _ in range(100):
        _m2 = mu_1.square() + Gamma_1.diag()
        w_1 = (xi / (_m2 + 1e-09)).sqrt()
        Gamma_2 = torch.linalg.inv(w_1.diag() + tau * A.t() @ A)
        mu_2 = tau * Gamma_2 @ A.t() @ y
    plt.plot(mu_2)
    plt.plot(x0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Takeaways

    1. The $w$-update equation that comes from MAP is pretty useless. It's sensitive to float/double precision
       and no amount of tinkering with $\xi$ resulted in a convergent algorithm.
    2. The $w$-update equation from EM (maximizing w.r.t $x$) is much better. It's less sensitive to precision
       and converges.
    3. As we want a distribution over $x$ *and* the EM-type $w$-update, we need to use VRLS.
    4. There does appear to be a sweet spot for $\xi$, which *suggests* we should try extended VRLS.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
