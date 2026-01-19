import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch

    import vrlsnmr.operators as op
    return op, plt


@app.cell
def _(op):
    ids = op.schedpg(m=50, n=1024)
    ids
    return (ids,)


@app.cell
def _(ids, plt):
    (_, ax) = plt.subplots()
    ax.bar(ids, height=1)
    ax.set_xlim((0, 1024))
    ax
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
