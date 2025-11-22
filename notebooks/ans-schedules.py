import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from functools import partial
    from itertools import product

    import matplotlib.pyplot as plt
    import torch

    import vrlsnmr.algorithms as algo
    from vrlsnmr.simulators import Signal
    return Signal, algo, partial, plt, product, torch


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
    return n, schedules


@app.cell
def _(n, plt, schedules, torch):
    n_pdf = n // 4

    pdf = schedules.mean(dim=1).narrow(dim=1, start=0, length=n_pdf).cpu()
    plt.bar(torch.arange(n_pdf), pdf[0])
    return


@app.cell
def _(n, schedules, torch):
    psf = (
        torch.fft.fft(schedules.cfloat(), dim=-1, norm="ortho")
        .roll(n // 4, dims=-1)
        .abs()
        .cpu()
    )
    return (psf,)


@app.cell
def _(n, plt, psf, torch):
    _x = torch.linspace(-0.5, 0.5, n // 2)
    _mean = psf[0].mean(dim=0)
    _std = psf[0].std(dim=0)

    (_, _ax) = plt.subplots()
    _ax.fill_between(
        x=_x,
        y1=_mean - 3 * _std,
        y2=_mean + 3 * _std,
        alpha=0.2,
    )
    _ax.plot(_x, _mean)
    _ax.set_xlabel("Frequency / a.u.")
    _ax.set_xlim((-0.5, 0.5))
    _ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0], labels=[])
    _ax.grid(alpha=0.2)

    _axin = _ax.inset_axes((0.65, 0.65, 0.3, 0.3))
    _axin.plot(_x, _mean)
    _axin.set_xlim((-0.1, 0.1))
    _axin.set_yticks([])

    _ax
    return


@app.cell
def _(n, plt, psf, torch):
    _x = torch.linspace(-0.5, 0.5, n // 2)
    _mean = psf[3].mean(dim=0)
    _std = psf[3].std(dim=0)

    (_, _ax) = plt.subplots()
    _ax.fill_between(
        x=_x,
        y1=_mean - 3 * _std,
        y2=_mean + 3 * _std,
        alpha=0.2,
    )
    _ax.plot(_x, _mean)
    _ax.set_xlim((-0.5, 0.5))
    _ax.set_xlabel("Frequency / a.u.")
    _ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0], labels=[])
    _ax.grid(alpha=0.2)

    _axin = _ax.inset_axes((0.65, 0.65, 0.3, 0.3))
    _axin.plot(_x, _mean)
    _axin.set_xlim((-0.1, 0.1))
    _axin.set_yticks([])

    _ax
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
