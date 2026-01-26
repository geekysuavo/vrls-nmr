import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch

    import vrlsnmr.operators as op
    return op, plt


@app.cell
def _():
    m = 50
    n = 1024
    return m, n


@app.cell
def _(m, n, op):
    ids_exp = op.schedexp(rate=0.01, m=m, n=n)
    ids_exp
    return (ids_exp,)


@app.cell
def _(m, n, op):
    ids_pg = op.schedpg(m=m, n=n)
    ids_pg
    return (ids_pg,)


@app.cell
def _(ids_exp, ids_pg, n, plt):
    (_, (left, right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    left.bar(ids_pg, height=1)
    left.set_xlim((0, n))
    left.set_title("PG")

    right.bar(ids_exp, height=1)
    right.set_xlim((0, n))
    right.set_title("Exp")

    left
    return


if __name__ == "__main__":
    app.run()
