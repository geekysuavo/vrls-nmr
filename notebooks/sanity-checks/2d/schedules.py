import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch

    import vrlsnmr.schedules as sch

    return plt, sch, torch


@app.cell
def _():
    m = 100
    n = (100, 100)
    return m, n


@app.cell
def _(m, n, plt, sch):
    ids_unif = sch.uniform(m=m, n=n)
    plt.scatter(*ids_unif.unbind(dim=1))
    return


@app.cell
def _(m, n, plt, sch):
    ids_exp = sch.exponential(rate=0.05, m=m, n=n)
    plt.scatter(*ids_exp.unbind(dim=1))
    return


@app.cell
def _(m, n, plt, sch):
    ids_pg = sch.poisson_gap(m=m, n=n)
    plt.scatter(*ids_pg.unbind(dim=1))
    return


@app.cell
def _(torch):
    torch.cartesian_prod(*([torch.arange(5)] * 3))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
