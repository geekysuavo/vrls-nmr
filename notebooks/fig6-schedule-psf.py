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
    x = torch.linspace(-0.5, 0.5, data.n // 2)

    i = 0
    sigma = data.sigma[i]
    mean = data.psf[i].mean(dim=0)
    std = data.psf[i].std(dim=0)

    (fig, (left, right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    left.fill_between(
        x=x,
        y1=mean - 3 * std,
        y2=mean + 3 * std,
        color=(0.9,) * 3,
    )
    left.plot(x, mean, color=(0.3,) * 3)
    left.set_xlabel("Frequency / a.u.")
    left.set_xlim((-0.5, 0.5))
    left.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0], labels=[])
    left.grid(color=(0.9,) * 3)
    left.text(
        -0.42, 1.85, "(a)",
        fontweight="bold",
        horizontalalignment="center",
    )

    inleft = left.inset_axes((0.65, 0.65, 0.3, 0.3))
    inleft.plot(x, mean, color=(0.3,) * 3)
    inleft.set_xlim((-0.1, 0.1))
    inleft.set_yticks([])

    # ====

    i = 3
    sigma = data.sigma[i]
    mean = data.psf[i].mean(dim=0)
    std = data.psf[i].std(dim=0)

    right.fill_between(
        x=x,
        y1=mean - 3 * std,
        y2=mean + 3 * std,
        color=(0.9,) * 3,
    )
    right.plot(x, mean, color=(0.3,) * 3)
    right.set_xlabel("Frequency / a.u.")
    right.set_xlim((-0.5, 0.5))
    right.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0], labels=[])
    right.grid(color=(0.9,) * 3)
    right.text(
        -0.42, 1.85, "(b)",
        fontweight="bold",
        horizontalalignment="center",
    )

    inright = right.inset_axes((0.65, 0.65, 0.3, 0.3))
    inright.plot(x, mean, color=(0.3,) * 3)
    inright.set_xlim((-0.1, 0.1))
    inright.set_yticks([])

    figure_path = Path.cwd() / "figures" / "figure-6.pdf"
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
