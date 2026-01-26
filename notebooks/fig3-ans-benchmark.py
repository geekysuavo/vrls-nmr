import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    return Path, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_parquet("ans-benchmark.parquet")
    return (df,)


@app.cell
def _(df, sns):
    # example of the kind of manips we'll need to do.
    _df = df.set_index(["identifier", "replicate", "region", "schedule"])
    _vrls = _df.query("algorithm == 'vrls'").error
    _ists = _df.query("algorithm == 'ists'").error
    _err = (_ists - _vrls) / _ists

    _df["rel_error_by_schedule"] = _err
    _df = _df.reset_index().query("algorithm == 'vrls'").drop(columns=["algorithm"])

    sns.histplot(x="error", hue="schedule", data=_df.query("region == 'noise'"))
    return


@app.cell
def _(df):
    agg = df.groupby(["sigma", "min_sparsity"])[["mse_noise_ans", "mse_noise_unif", "mse_noise_diff"]].mean()
    agg
    return


@app.cell
def _(Path, df, plt, sns):
    (fig, ax) = plt.subplots(figsize=(6, 4))

    sns.barplot(
        x="sigma",
        y="mse_noise_diff",
        hue="min_sparsity",
        data=df,
        ax=ax,
        palette=[
            (0.8,) * 3,
            (0.6,) * 3,
            (0.4,) * 3,
        ]
    )

    ax.set_xlabel(r"Measurement noise ($\sigma$)")
    ax.set_ylabel("Relative error reduction")
    ax.grid(color=(0.9,) * 3)
    ax.set_axisbelow(True)

    legend = ax.get_legend()
    legend.set_title("Undersampling")
    legend.texts[0].set_text("5%")
    legend.texts[1].set_text("10%")
    legend.texts[2].set_text("20%")

    figure_path = Path.cwd() / "figures" / "figure-3.pdf"
    figure_path.parent.mkdir(exist_ok=True)
    plt.savefig(
        figure_path,
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    fig
    return


@app.cell
def _(mask, mu, mu_unif, plt, x0):
    (_, _ax) = plt.subplots()

    _ax.plot(x0.real.cpu())
    _ax.plot(mu.real.cpu())
    _ax.plot(mu_unif.real.cpu())
    _ax.plot(mask.cpu())
    _ax.grid(alpha=0.2)

    _ax.set_xlim((900, 1024))
    #_ax.set_ylim((-1, 1))

    _ax
    return


if __name__ == "__main__":
    app.run()
