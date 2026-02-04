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
def _(algo, ground_truth, partial):
    sigma = 0.1
    niter = 100
    n = 2048
    m = 64

    (_, ids, *_) = algo.ans(
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
    return ids, sigma


@app.cell
def _(ground_truth, ids, sigma, torch):
    n_td = 512

    x = torch.arange(n_td)
    y = ground_truth(x.cuda()).cpu()

    x_meas = ids.cpu()
    y_meas = ground_truth(ids, noise=sigma).cpu()
    return x, x_meas, y, y_meas


@app.cell
def _(Path, plt, sigma, x, x_meas, y, y_meas):
    (fig, (top, bottom)) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

    top.fill_between(x, y.real - sigma, y.real + sigma, color="lightblue")
    top.plot(x, y.real, linewidth=1)
    top.scatter(x_meas, y_meas.real, s=10)
    #top.set_xlabel("Grid index")
    top.set_ylabel("Signal (real)")
    top.grid(color=(0.9,) * 3)

    bottom.fill_between(x, y.imag - sigma, y.imag + sigma, color="lightgreen")
    bottom.plot(x, y.imag, linewidth=1, color="darkgreen")
    bottom.scatter(x_meas, y_meas.imag, s=10, color="darkgreen")
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
