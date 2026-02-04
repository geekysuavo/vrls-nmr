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
    return Path, np, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_parquet("ans-benchmark.parquet")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df["nmse"] = df.sumsq / df.norm
    df["rsq"] = 1 - df.nmse
    return


@app.cell
def _(np, pd):
    def relative_error_by_algorithm(df: pd.DataFrame) -> pd.DataFrame:
        dfs = dict(iter(df.groupby("schedule", sort=False, group_keys=False)))
        nmse = {
            group: group_df.nmse.reset_index(drop=True)
            for group, group_df in dfs.items()
        }
        delta = {
            group: group_nmse - nmse["ans"]
            for group, group_nmse in nmse.items()
        }
        error = {
            group: (group_nmse - nmse["ans"]) / group_nmse
            for group, group_nmse in nmse.items()
        }
        delta["ans"] *= np.nan

        for group, group_delta in delta.items():
            group_delta.index = dfs[group].index

        for group, group_err in error.items():
            group_err.index = dfs[group].index

        new_df = df.copy(deep=False)
        all_delta = pd.concat(delta.values())
        all_error = pd.concat(error.values())
        new_df["rel_error_by_algorithm"] = all_error
        new_df["delta_nmse_by_algorithm"] = all_delta
        new_df["is_improved_by_algorithm"] = all_delta.ge(0)
        return new_df
    return (relative_error_by_algorithm,)


@app.cell
def _(np, pd):
    def relative_error_by_schedule(df: pd.DataFrame) -> pd.DataFrame:
        dfs = dict(iter(df.groupby("algorithm", sort=False, group_keys=False)))
        nmse = {
            group: group_df.nmse.reset_index(drop=True)
            for group, group_df in dfs.items()
        }
        delta = {
            group: group_nmse - nmse["vrls"]
            for group, group_nmse in nmse.items()
        }
        error = {
            group: (group_nmse - nmse["vrls"]) / group_nmse
            for group, group_nmse in nmse.items()
        }
        delta["vrls"] *= np.nan

        for group, group_delta in delta.items():
            group_delta.index = dfs[group].index

        for group, group_err in error.items():
            group_err.index = dfs[group].index

        new_df = df.copy(deep=False)
        all_delta = pd.concat(delta.values())
        all_error = pd.concat(error.values())
        new_df["rel_error_by_schedule"] = all_error
        new_df["delta_nmse_by_schedule"] = all_delta
        new_df["is_improved_by_schedule"] = all_delta.ge(0)
        return new_df
    return (relative_error_by_schedule,)


@app.cell
def _(df, relative_error_by_algorithm, relative_error_by_schedule):
    _df = df

    _columns = ["sigma", "num_components", "amplitude_dof", "min_sparsity", "algorithm"]
    _df = _df.groupby(_columns, sort=False).apply(relative_error_by_algorithm).reset_index()

    _columns = ["sigma", "num_components", "amplitude_dof", "min_sparsity", "schedule"]
    _df = _df.groupby(_columns, sort=False).apply(relative_error_by_schedule).reset_index()

    full_df = _df
    return (full_df,)


@app.cell
def _(full_df):
    full_df
    return


@app.cell
def _(full_df, pd):
    def _fraction(region: str, schedule: str, min_sparsity: float) -> float:
        _query = (
            f"region == '{region}' "
            f"and schedule == '{schedule}' "
            f"and min_sparsity == {min_sparsity} "
            f"and algorithm == 'vrls'"
        )
        return full_df.query(_query).is_improved_by_algorithm.mean().item()

    _data = [
        dict(
            region=region,
            schedule=schedule,
            min_sparsity=min_sparsity,
            fraction=_fraction(region, schedule, min_sparsity),
        )
        for region in ("noise", "signal", "total")
        for schedule in ("unif", "exp", "pg")
        for min_sparsity in (0.05, 0.1, 0.2)
    ]

    fraction_df = pd.DataFrame(_data)
    return (fraction_df,)


@app.cell
def _(fraction_df):
    fraction_df
    return


@app.cell
def _(Path, fraction_df, full_df, plt, sns):
    (fig, (left, right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.barplot(
        x="sigma",
        y="delta_nmse_by_algorithm",
        hue="min_sparsity",
        data=full_df.query(
            "region == 'noise'"
            " and schedule == 'unif'"
            " and algorithm == 'vrls'"
        ),
        ax=left,
        palette=[
            (0.8,) * 3,
            (0.6,) * 3,
            (0.4,) * 3,
        ]
    )

    sns.barplot(
        x="schedule",
        y="fraction",
        hue="region",
        data=fraction_df,
        ax=right,
        palette=[
            (0.8,) * 3,
            (0.6,) * 3,
            (0.4,) * 3,
        ]
    )

    left.set_xlabel(r"Measurement noise ($\sigma$)")
    left.set_ylabel(r"$\Delta$NMSE$_{noise}}$")
    left.grid(color=(0.9,) * 3)
    left.set_axisbelow(True)
    left.text(
        0, 3.7, "(a)",
        fontweight="bold",
        horizontalalignment="center",
    )

    right.set_xlabel("Schedule")
    right.set_ylabel("Probability of improvement")
    right.grid(color=(0.9,) * 3)
    right.set_axisbelow(True)
    right.text(
        0, 0.95, "(b)",
        fontweight="bold",
        horizontalalignment="center",
    )

    left_legend = left.get_legend()
    left_legend.set_title("Undersampling")
    left_legend.texts[0].set_text("5%")
    left_legend.texts[1].set_text("10%")
    left_legend.texts[2].set_text("20%")

    right_legend = right.get_legend()
    right_legend.set_title("Spectral region")
    right_legend.texts[0].set_text("Noise")
    right_legend.texts[1].set_text("Signal")
    right_legend.texts[2].set_text("Total")

    right.set_xticks(["unif", "exp", "pg"])
    right.set_xticklabels(["Uniform", "Exponential", "Poisson-gap"])

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
    return (figure_path,)


@app.cell
def _(figure_path, full_df, plt, sns):
    (_fig, _ax) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    _kwargs = dict(
        x="delta_nmse_by_schedule",
        hue="schedule",
        element="step",
    )

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'noise'"
            f" and algorithm == 'ists'"
        ),
        ax=_ax[0, 0],
    )
    _ax[0, 0].set_xlabel(r"$\Delta$NMSE$_{noise}$ (IST - VRLS)")
    _ax[0, 0].grid(color=(0.9,) * 3)
    _ax[0, 0].set_axisbelow(True)
    _ax[0, 0].set_xlim((-5, 20))

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'signal'"
            " and algorithm == 'ists'"
        ),
        ax=_ax[0, 1],
        legend=False,
    )
    _ax[0, 1].set_xlabel(r"$\Delta$NMSE$_{signal}$ (IST - VRLS)")
    _ax[0, 1].grid(color=(0.9,) * 3)
    _ax[0, 1].set_axisbelow(True)
    _ax[0, 1].set_xlim((-0.5, 1))

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'noise'"
            f" and algorithm == 'fmf'"
        ),
        ax=_ax[1, 0],
        legend=False,
    )
    _ax[1, 0].set_xlabel(r"$\Delta$NMSE$_{noise}$ (FMF - VRLS)")
    _ax[1, 0].grid(color=(0.9,) * 3)
    _ax[1, 0].set_axisbelow(True)
    _ax[1, 0].set_xlim((-5, 20))

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'signal'"
            " and algorithm == 'fmf'"
        ),
        ax=_ax[1, 1],
        legend=False,
    )
    _ax[1, 1].set_xlabel(r"$\Delta$NMSE$_{signal}$ (FMF - VRLS)")
    _ax[1, 1].grid(color=(0.9,) * 3)
    _ax[1, 1].set_axisbelow(True)
    _ax[1, 1].set_xlim((-0.5, 1))

    _legend = _ax[0, 0].get_legend()
    _legend.set_title("Schedule")
    _legend.texts[0].set_text("ANS")
    _legend.texts[1].set_text("Uniform")
    _legend.texts[2].set_text("Exponential")
    _legend.texts[3].set_text("Poisson-gap")

    plt.savefig(
        figure_path.parent / "figure-s1.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    _fig
    return


@app.cell
def _(figure_path, full_df, plt, sns):
    (_fig, _ax) = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    _kwargs = dict(
        x="delta_nmse_by_algorithm",
        hue="algorithm",
        element="step",
    )

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'noise'"
            f" and schedule == 'unif'"
        ),
        ax=_ax[0, 0],
    )
    _ax[0, 0].set_xlabel(r"$\Delta$NMSE$_{noise}$ (Uniform - ANS)")
    _ax[0, 0].grid(color=(0.9,) * 3)
    _ax[0, 0].set_axisbelow(True)
    _ax[0, 0].set_xlim((-20, 20))

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'signal'"
            " and schedule == 'unif'"
        ),
        ax=_ax[0, 1],
        legend=False,
    )
    _ax[0, 1].set_xlabel(r"$\Delta$NMSE$_{signal}$ (Uniform - ANS)")
    _ax[0, 1].grid(color=(0.9,) * 3)
    _ax[0, 1].set_axisbelow(True)
    _ax[0, 1].set_xlim((-1.25, 0.6))

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'noise'"
            f" and schedule == 'exp'"
        ),
        ax=_ax[1, 0],
        legend=False,
    )
    _ax[1, 0].set_xlabel(r"$\Delta$NMSE$_{noise}$ (Exponential - ANS)")
    _ax[1, 0].grid(color=(0.9,) * 3)
    _ax[1, 0].set_axisbelow(True)
    _ax[1, 0].set_xlim((-20, 20))

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'signal'"
            " and schedule == 'exp'"
        ),
        ax=_ax[1, 1],
        legend=False,
    )
    _ax[1, 1].set_xlabel(r"$\Delta$NMSE$_{signal}$ (Exponential - ANS)")
    _ax[1, 1].grid(color=(0.9,) * 3)
    _ax[1, 1].set_axisbelow(True)
    _ax[1, 1].set_xlim((-1.25, 0.6))

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'noise'"
            f" and schedule == 'pg'"
        ),
        ax=_ax[2, 0],
        legend=False,
    )
    _ax[2, 0].set_xlabel(r"$\Delta$NMSE$_{noise}$ (Poisson-gap - ANS)")
    _ax[2, 0].grid(color=(0.9,) * 3)
    _ax[2, 0].set_axisbelow(True)
    _ax[2, 0].set_xlim((-20, 20))

    sns.histplot(
        **_kwargs,
        data=full_df.query(
            "region == 'signal'"
            " and schedule == 'pg'"
        ),
        ax=_ax[2, 1],
        legend=False,
    )
    _ax[2, 1].set_xlabel(r"$\Delta$NMSE$_{signal}$ (Poisson-gap - ANS)")
    _ax[2, 1].grid(color=(0.9,) * 3)
    _ax[2, 1].set_axisbelow(True)
    _ax[2, 1].set_xlim((-1.25, 0.6))

    _legend = _ax[0, 0].get_legend()
    _legend.set_title("Algorithm")
    _legend.texts[0].set_text("VRLS")
    _legend.texts[1].set_text("FMF-VRLS")
    _legend.texts[2].set_text("IST-S")

    plt.savefig(
        figure_path.parent / "figure-s2.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    _fig
    return


if __name__ == "__main__":
    app.run()
