import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import itertools
    import time

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch
    import torch.nn.functional as F

    from vrlsnmr.algorithms import vrls, vrls_mf
    from vrlsnmr.simulators import Signal
    return F, Signal, itertools, pd, plt, sns, time, torch, vrls, vrls_mf


@app.cell
def _(Signal, torch):
    ground_truth = Signal(
        frequencies=torch.tensor([-0.083, -0.026, 0.038, 0.042, 0.074, 0.091]),
        decayrates=torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
        amplitudes=torch.tensor([1.0, 1.0, 0.1, 0.3, 1.0, 0.9]),
        phases=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    return (ground_truth,)


@app.cell
def _():
    sched = [
        1, 2, 3, 5, 6, 8, 9, 11, 13, 16,
        17, 19, 21, 25, 29, 33, 38, 46, 56, 58,
        69, 73, 82, 98, 116, 118, 138, 165, 187, 199,
        217, 235, 267, 310, 365, 408, 415, 457, 475, 510,
        595, 615, 648, 660, 667, 738, 851, 872, 873, 959,
        991,
    ]
    return (sched,)


@app.cell
def _(ground_truth, sched, torch):
    sigma = 0.1
    tau = 1 / sigma**2
    xi = tau
    n = 2048
    niter = 1000

    ids = torch.tensor(sched) - 1
    y = ground_truth(ids, noise=sigma)

    ids = ids.cuda()
    y = y.cuda()
    return ids, n, niter, sigma, tau, xi, y


@app.cell
def _(ids, n, niter, tau, vrls, vrls_mf, xi, y):
    (mu, gamma_diag, yhat, sigma_diag) = vrls(y, ids, tau, xi, n, niter // 10)
    (mu_fast, gamma_fast, _, _) = vrls_mf(y, ids, tau, xi, n, niter)

    mu = mu.roll(n // 2).squeeze(dim=0)
    mu_fast = mu_fast.roll(n // 2).squeeze(dim=0)
    gamma_diag = gamma_diag.roll(n // 2).squeeze(dim=0)
    gamma_fast = gamma_fast.roll(n // 2).squeeze(dim=0)
    return mu, mu_fast


@app.cell
def _(F, ground_truth, mu, mu_fast, n, plt, sigma, torch):
    t = torch.arange(n // 2)
    spect = torch.fft.fft(
        F.pad(ground_truth(t, noise=sigma), (0, n // 2)),
        norm="ortho",
    ).roll(n // 2)

    plt.plot(mu.real.cpu())
    plt.xlim((800, 1000))
    plt.plot(mu_fast.real.cpu())
    #plt.plot(spect.real.cpu(), alpha=0.5)
    return


@app.cell
def _(
    ground_truth,
    itertools,
    mo,
    niter,
    pd,
    sigma,
    tau,
    time,
    torch,
    vrls,
    vrls_mf,
    xi,
):
    data = []

    bs_values = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    n_values = (64, 128, 256, 512, 1024, 2048)
    m_values = (16, 32, 64)

    for _bs, _n, _m in mo.status.progress_bar(
        itertools.product(bs_values, n_values, m_values),
        total=len(bs_values) * len(n_values) * len(m_values),
    ):
        if _m >= _n:
            continue

        _ids = torch.randperm(_n).narrow(dim=0, start=0, length=_m)
        _y = ground_truth(_ids, noise=sigma).expand(_bs, -1).cuda()
        _ids = _ids.cuda()

        for rep in range(3):
            t0 = time.perf_counter()
            _ = vrls(_y, _ids, tau, xi, _n, niter // 10)
            t1 = time.perf_counter()
            _ = vrls_mf(_y, _ids, tau, xi, _n, niter)
            t2 = time.perf_counter()

            datum = dict(bs=_bs, m=_m, n=_n, trial=rep, Algorithm="VRLS", time=t1 - t0)
            data.append(datum)

            datum = dict(bs=_bs, m=_m, n=_n, trial=rep, Algorithm="FMF-VRLS", time=t2 - t1)
            data.append(datum)

    df = pd.DataFrame(data)
    df
    return (df,)


@app.cell
def _(df, sns):
    ax = sns.lineplot(x="bs", y="time", hue="Algorithm", style="m", data=df)
    ax.grid(alpha=0.2)
    ax.set_xlabel("Number of parallel reconstructions")
    ax.set_ylabel("Reconstruction time")
    ax.set_yscale("log")
    ax
    return


if __name__ == "__main__":
    app.run()
