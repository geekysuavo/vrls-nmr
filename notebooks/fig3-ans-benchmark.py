import marimo

__generated_with = "0.19.6"
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
def _(np, pd):
    def relative_error_by_algorithm(df: pd.DataFrame) -> pd.DataFrame:
        dfs = dict(iter(df.groupby("schedule", sort=False, group_keys=False)))
        errors = {
            group: group_df.error.reset_index(drop=True)
            for group, group_df in dfs.items()
        }
        error_unif = (errors["unif"] - errors["ans"]) / errors["unif"]
        error_exp = (errors["exp"] - errors["ans"]) / errors["exp"]
        error_pg = (errors["pg"] - errors["ans"]) / errors["pg"]
        error_ans = error_pg * np.nan

        error_ans.index = dfs["ans"].index
        error_unif.index = dfs["unif"].index
        error_exp.index = dfs["exp"].index
        error_pg.index = dfs["pg"].index

        new_df = df.copy(deep=False)
        error = pd.concat((error_ans, error_unif, error_exp, error_pg))
        new_df["rel_error_by_algorithm"] = error
        return new_df
    return (relative_error_by_algorithm,)


@app.cell
def _(np, pd):
    def relative_error_by_schedule(df: pd.DataFrame) -> pd.DataFrame:
        dfs = dict(iter(df.groupby("algorithm", sort=False, group_keys=False)))
        errors = {
            group: group_df.error.reset_index(drop=True)
            for group, group_df in dfs.items()
        }
        error_ists = (errors["ists"] - errors["vrls"]) / errors["ists"]
        error_fmf = (errors["fmf"] - errors["vrls"]) / errors["fmf"]
        error_vrls = error_fmf * np.nan

        error_vrls.index = dfs["vrls"].index
        error_ists.index = dfs["ists"].index
        error_fmf.index = dfs["fmf"].index

        new_df = df.copy(deep=False)
        error = pd.concat((error_vrls, error_ists, error_fmf))
        new_df["rel_error_by_schedule"] = error
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
def _(Path, full_df, plt, sns):
    (fig, ax) = plt.subplots(figsize=(6, 4))

    sns.barplot(
        x="sigma",
        y="rel_error_by_algorithm",
        hue="min_sparsity",
        data=full_df.query(
            "region == 'noise'"
            " and schedule == 'unif'"
            " and algorithm == 'vrls'"
        ),
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
    return (figure_path,)


@app.cell
def _(figure_path, full_df, plt, sns):
    (_fig, (_left, _right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.histplot(
        x="rel_error_by_schedule",
        hue="schedule",
        element="step",
        data=full_df.query(
            "region == 'noise'"
            " and algorithm == 'ists'"
        ),
        ax=_left,
    )

    sns.histplot(
        x="rel_error_by_schedule",
        hue="schedule",
        element="step",
        data=full_df.query(
            "region == 'signal'"
            " and algorithm == 'ists'"
        ),
        ax=_right,
        legend=False,
    )

    _left.set_title("VRLS vs. IST-S / Noise region only")
    _left.set_xlabel("Relative error reduction")
    _left.grid(color=(0.9,) * 3)
    _left.set_axisbelow(True)

    _right.set_title("VRLS vs. IST-S / Signal region only")
    _right.set_xlabel("Relative error reduction")
    _right.grid(color=(0.9,) * 3)
    _right.set_axisbelow(True)
    _right.set_xlim((-0.8, 0.8))

    _legend = _left.get_legend()
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
    (_fig, (_left, _right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.histplot(
        x="rel_error_by_schedule",
        hue="schedule",
        element="step",
        data=full_df.query(
            "region == 'noise'"
            " and algorithm == 'fmf'"
        ),
        ax=_left,
    )

    sns.histplot(
        x="rel_error_by_schedule",
        hue="schedule",
        element="step",
        data=full_df.query(
            "region == 'signal'"
            " and algorithm == 'fmf'"
        ),
        ax=_right,
        legend=False,
    )

    _left.set_title("VRLS vs. FMF-VRLS / Noise region only")
    _left.set_xlabel("Relative error reduction")
    _left.grid(color=(0.9,) * 3)
    _left.set_axisbelow(True)

    _right.set_title("VRLS vs. FMF-VRLS / Signal region only")
    _right.set_xlabel("Relative error reduction")
    _right.grid(color=(0.9,) * 3)
    _right.set_axisbelow(True)
    _right.set_xlim((-1, 1))

    _legend = _left.get_legend()
    _legend.set_title("Schedule")
    _legend.texts[0].set_text("ANS")
    _legend.texts[1].set_text("Uniform")
    _legend.texts[2].set_text("Exponential")
    _legend.texts[3].set_text("Poisson-gap")

    plt.savefig(
        figure_path.parent / "figure-s2.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    _fig
    return


@app.cell
def _(figure_path, full_df, plt, sns):
    (_fig, (_left, _right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.histplot(
        x="rel_error_by_algorithm",
        hue="algorithm",
        element="step",
        data=full_df.query(
            "region == 'noise'"
            " and schedule == 'unif'"
        ),
        ax=_left,
    )

    sns.histplot(
        x="rel_error_by_algorithm",
        hue="algorithm",
        element="step",
        data=full_df.query(
            "region == 'signal'"
            " and schedule == 'unif'"
        ),
        ax=_right,
        legend=False,
    )

    _left.set_title("ANS vs. Uniform / Noise region only")
    _left.set_xlabel("Relative error reduction")
    _left.grid(color=(0.9,) * 3)
    _left.set_axisbelow(True)

    _right.set_title("ANS vs. Uniform / Signal region only")
    _right.set_xlabel("Relative error reduction")
    _right.grid(color=(0.9,) * 3)
    _right.set_axisbelow(True)
    #_right.set_xlim((-1, 1))

    _legend = _left.get_legend()
    _legend.set_title("Algorithm")
    _legend.texts[0].set_text("VRLS")
    _legend.texts[1].set_text("FMF-VRLS")
    _legend.texts[2].set_text("IST-S")

    plt.savefig(
        figure_path.parent / "figure-s3.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    _fig
    return


@app.cell
def _(figure_path, full_df, plt, sns):
    (_fig, (_left, _right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.histplot(
        x="rel_error_by_algorithm",
        hue="algorithm",
        element="step",
        data=full_df.query(
            "region == 'noise'"
            " and schedule == 'exp'"
        ),
        ax=_left,
    )

    sns.histplot(
        x="rel_error_by_algorithm",
        hue="algorithm",
        element="step",
        data=full_df.query(
            "region == 'signal'"
            " and schedule == 'exp'"
        ),
        ax=_right,
        legend=False,
    )

    _left.set_title("ANS vs. Exponential / Noise region only")
    _left.set_xlabel("Relative error reduction")
    _left.grid(color=(0.9,) * 3)
    _left.set_axisbelow(True)
    _left.set_xlim((-15, 1))

    _right.set_title("ANS vs. Exponential / Signal region only")
    _right.set_xlabel("Relative error reduction")
    _right.grid(color=(0.9,) * 3)
    _right.set_axisbelow(True)
    #_right.set_xlim((-1, 1))

    _legend = _left.get_legend()
    _legend.set_title("Algorithm")
    _legend.texts[0].set_text("VRLS")
    _legend.texts[1].set_text("FMF-VRLS")
    _legend.texts[2].set_text("IST-S")

    plt.savefig(
        figure_path.parent / "figure-s4.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    _fig
    return


@app.cell
def _(figure_path, full_df, plt, sns):
    (_fig, (_left, _right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.histplot(
        x="rel_error_by_algorithm",
        hue="algorithm",
        element="step",
        data=full_df.query(
            "region == 'noise'"
            " and schedule == 'pg'"
        ),
        ax=_left,
    )

    sns.histplot(
        x="rel_error_by_algorithm",
        hue="algorithm",
        element="step",
        data=full_df.query(
            "region == 'signal'"
            " and schedule == 'pg'"
        ),
        ax=_right,
        legend=False,
    )

    _left.set_title("ANS vs. Poisson-gap / Noise region only")
    _left.set_xlabel("Relative error reduction")
    _left.grid(color=(0.9,) * 3)
    _left.set_axisbelow(True)
    _left.set_xlim((-20, 1))

    _right.set_title("ANS vs. Poisson-gap / Signal region only")
    _right.set_xlabel("Relative error reduction")
    _right.grid(color=(0.9,) * 3)
    _right.set_axisbelow(True)
    _right.set_xlim((-2, 0.5))

    _legend = _left.get_legend()
    _legend.set_title("Algorithm")
    _legend.texts[0].set_text("VRLS")
    _legend.texts[1].set_text("FMF-VRLS")
    _legend.texts[2].set_text("IST-S")

    plt.savefig(
        figure_path.parent / "figure-s5.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    _fig
    return


if __name__ == "__main__":
    app.run()
