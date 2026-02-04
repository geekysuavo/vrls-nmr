import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    return pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_parquet("ans-benchmark.parquet")
    return (df,)


@app.cell
def _(df):
    df["nmse"] = df.sumsq / df.norm
    df["rsq"] = 1 - df.nmse
    return


@app.cell
def _(df, plt, sns):
    (_fig, _ax) = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))
    _kwargs = dict(
        x="nmse",
        hue="schedule",
        element="step",
    )
    _rest = "algorithm == 'vrls'"

    sns.histplot(
        **_kwargs,
        data=df.query(f"region == 'noise' and {_rest}"),
        ax=_ax[0],
    )
    _ax[0].set_xlabel(r"NMSE$_{noise}$ (VRLS)")
    _ax[0].grid(color=(0.9,) * 3)
    _ax[0].set_axisbelow(True)
    _ax[0].set_xlim((0, 5))

    sns.histplot(
        **_kwargs,
        data=df.query(f"region == 'signal' and {_rest}"),
        ax=_ax[1],
        legend=False,
    )
    _ax[1].set_xlabel(r"NMSE$_{signal}$ (VRLS)")
    _ax[1].grid(color=(0.9,) * 3)
    _ax[1].set_axisbelow(True)

    sns.histplot(
        **_kwargs,
        data=df.query(f"region == 'total' and {_rest}"),
        ax=_ax[2],
        legend=False,
    )
    _ax[2].set_xlabel(r"NMSE$_{total}$ (VRLS)")
    _ax[2].grid(color=(0.9,) * 3)
    _ax[2].set_axisbelow(True)

    _legend = _ax[0].get_legend()
    _legend.set_title("Schedule")
    _legend.texts[0].set_text("ANS")
    _legend.texts[1].set_text("Uniform")
    _legend.texts[2].set_text("Exponential")
    _legend.texts[3].set_text("Poisson-gap")

    _fig
    return


if __name__ == "__main__":
    app.run()
