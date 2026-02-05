import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch

    import vrlsnmr.algorithms as algo
    from vrlsnmr.simulators import Signal
    return Path, Signal, algo, partial, plt, torch


@app.cell
def _(Signal, torch):
    ground_truth = Signal(
        frequencies=torch.tensor([-0.083, -0.026, 0.038, 0.042, 0.074, 0.091]),
        decayrates=torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
        amplitudes=torch.tensor([1.0, 1.0, 0.1, 0.3, 1.0, 0.9]),
        phases=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    ground_truth.cuda()
    return (ground_truth,)


@app.cell
def _(algo, ground_truth, partial, torch):
    sigma = 0.1
    niter = 100
    n = 2048
    m = 64

    (yobs, xobs, _, _, yhat, yvar) = algo.ans(
        model=algo.vrls,
        measure=partial(ground_truth, noise=sigma),
        m_initial=8,
        n_initial=32,
        m_final=m,
        n_final=n,
        min_sparsity=0.1,
        tau=1 / sigma**2,
        xi=1 / sigma**2,
        niter=niter,
    )

    xobs = xobs.cpu()
    yobs = yobs.cpu()
    yhat = yhat.squeeze(dim=0).cpu()
    yvar = yvar.squeeze(dim=0).cpu()

    n_td = yhat.size(dim=0)
    x = torch.arange(n_td)
    y0 = ground_truth(x.cuda()).cpu()
    return x, xobs, y0, yhat, yobs, yvar


@app.cell
def _(Path, plt, x, xobs, y0, yhat, yobs, yvar):
    (fig, (top, bottom)) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

    top.fill_between(
        x,
        yhat.real - yvar.sqrt(),
        yhat.real + yvar.sqrt(),
        color="lightblue",
    )
    top.plot(x, y0.real, linewidth=1, color="grey", linestyle="--")
    top.plot(x, yhat.real, linewidth=1)
    top.scatter(xobs, yobs.real, s=10)
    top.set_ylabel("Signal (real)")
    top.grid(color=(0.9,) * 3)

    bottom.fill_between(
        x,
        yhat.imag - yvar.sqrt(),
        yhat.imag + yvar.sqrt(),
        color="lightgreen",
    )
    bottom.plot(x, y0.imag, linewidth=1, color="grey", linestyle="--")
    bottom.plot(x, yhat.imag, linewidth=1, color="darkgreen")
    bottom.scatter(xobs, yobs.imag, s=10, color="darkgreen")
    bottom.set_xlabel("Grid index")
    bottom.set_ylabel("Signal (imaginary)")
    bottom.grid(color=(0.9,) * 3)

    figure_path = Path.cwd() / "figures" / "figure-s3.pdf"
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
