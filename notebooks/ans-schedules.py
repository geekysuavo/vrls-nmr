import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from functools import partial
    from itertools import product
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch

    import vrlsnmr.algorithms as algo
    from vrlsnmr.simulators import Signal
    return Path, Signal, algo, partial, plt, product, torch


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
def _(algo, ground_truth, mo, partial, product, torch):
    m = 64
    n = 2048
    niter = 100

    num_replicates = 1000
    sigma_values = (0.01, 0.05, 0.1, 0.5)

    schedules = torch.zeros(
        (len(sigma_values), num_replicates, n // 2),
        dtype=torch.float,
        device="cuda",
    )

    for (index, sigma), rep in mo.status.progress_bar(
        product(enumerate(sigma_values), range(num_replicates)),
        total=len(sigma_values) * num_replicates,
    ):
        (_, ids, *_) = algo.ans(
            model=algo.vrls,
            measure=partial(ground_truth, noise=sigma),
            m_initial=16,
            n_initial=64,
            m_final=m,
            n_final=n,
            min_sparsity=0.1,
            tau=1 / sigma**2,
            xi=1 / sigma**2,
            niter=niter,
        )

        sched = schedules.new_zeros(n // 2)
        sched[ids] = 1
        schedules[index, rep] = sched
    return n, schedules, sigma_values


@app.cell
def _(n, schedules, torch):
    n_pdf = n // 4

    pdf = schedules.mean(dim=1).narrow(dim=1, start=0, length=n_pdf).cpu()

    psf = (
        torch.fft.fft(schedules.cfloat(), dim=-1, norm="ortho")
        .roll(n_pdf, dims=-1)
        .abs()
        .cpu()
    )
    return n_pdf, pdf, psf


@app.cell
def _(Path, n_pdf, pdf, plt, sigma_values, torch):
    _i = 0
    _sigma = sigma_values[_i]
    _x = torch.arange(n_pdf)
    _y = pdf[_i]

    (_fig, ax) = plt.subplots(figsize=(6, 4))

    ax.bar(_x, _y, linewidth=2, color=(0,) * 3)
    ax.set_xlabel("Grid index")
    ax.set_ylabel("Selection frequency")
    ax.grid(color=(0.9,) * 3)

    plt.savefig(
        Path.cwd() / "figure-4.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    _fig
    return


@app.cell
def _(Path, n, plt, psf, sigma_values, torch):
    _x = torch.linspace(-0.5, 0.5, n // 2)

    _i = 0
    _sigma = sigma_values[_i]
    _mean = psf[_i].mean(dim=0)
    _std = psf[_i].std(dim=0)

    (_fig, (left, right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    left.fill_between(
        x=_x,
        y1=_mean - 3 * _std,
        y2=_mean + 3 * _std,
        color=(0.9,) * 3,
    )
    left.plot(_x, _mean, color=(0.3,) * 3)
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
    inleft.plot(_x, _mean, color=(0.3,) * 3)
    inleft.set_xlim((-0.1, 0.1))
    inleft.set_yticks([])

    # ====

    _i = 3
    _sigma = sigma_values[_i]
    _mean = psf[_i].mean(dim=0)
    _std = psf[_i].std(dim=0)

    right.fill_between(
        x=_x,
        y1=_mean - 3 * _std,
        y2=_mean + 3 * _std,
        color=(0.9,) * 3,
    )
    right.plot(_x, _mean, color=(0.3,) * 3)
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
    inright.plot(_x, _mean, color=(0.3,) * 3)
    inright.set_xlim((-0.1, 0.1))
    inright.set_yticks([])

    plt.savefig(
        Path.cwd() / "figure-5.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    _fig
    return


if __name__ == "__main__":
    app.run()
