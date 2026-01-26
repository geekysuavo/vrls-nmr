import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import pickle

    import matplotlib.pyplot as plt
    import torch
    return Path, pickle, plt, torch


@app.cell
def _(pickle):
    with open("ans-schedules.pickle", "rb") as file:
        data = pickle.load(file)
    return (data,)


@app.cell
def _(Path, data, plt, torch):
    i = 0
    sigma = data.sigma[i]
    x = torch.arange(data.n_pdf)
    y = data.pdf[i]

    (fig, ax) = plt.subplots(figsize=(6, 4))

    ax.bar(x, y, linewidth=2, color=(0,) * 3)
    ax.set_xlabel("Grid index")
    ax.set_ylabel("Selection frequency")
    ax.grid(color=(0.9,) * 3)

    figure_path = Path.cwd() / "figures" / "figure-5.pdf"
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


if __name__ == "__main__":
    app.run()
