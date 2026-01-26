import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from functools import partial
    from itertools import product
    import pickle
    from types import SimpleNamespace

    import torch

    import vrlsnmr.algorithms as algo
    from vrlsnmr.simulators import Signal
    return Signal, SimpleNamespace, algo, partial, pickle, product, torch


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
def _(SimpleNamespace, n, n_pdf, pdf, pickle, psf, sigma_values):
    data = SimpleNamespace(
        n=n,
        n_pdf=n_pdf,
        pdf=pdf,
        psf=psf,
        sigma=sigma_values,
    )

    with open("ans-schedules.pickle", "wb") as file:
        pickle.dump(data, file)
    return


if __name__ == "__main__":
    app.run()
