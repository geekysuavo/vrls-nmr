import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import math
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    import vrlsnmr.algorithms as algo
    return Path, algo, plt, torch


@app.function
def parse(line: str) -> tuple[int, int, float, float]:
    (i, j, a, b) = line.strip().split()
    return (int(i), int(j), float(a), float(b))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Command to partially process the `ser` file
    ```bash
    hx -i . \
       -f resize[1]:size=1024 \
       -f fft[1] \
       -f phase[1]:ph0=45,ph1=-85 \
       -f real[1] \
       -F text -o proc.dat
    ```
    """)
    return


@app.cell
def _(Path):
    proc_dir = Path.home() / "camera" / "data" / "nhsqc-gb1"

    data_file = proc_dir / "proc.dat"
    assert data_file.is_file()

    sched_file = proc_dir / "nuslist"
    assert sched_file.is_file()

    with data_file.open() as fh:
        data = [
            parse(line)
            for line in fh
            if not line.startswith("#")
        ]

    with sched_file.open() as fh:
        sched = [int(line.strip()) for line in fh]
    return data, sched


@app.cell
def _(data, torch):
    td_shape = (40, 1024)
    td_scale = 1.0e6

    real = -torch.tensor([d[2] for d in data]).div(td_scale).view(td_shape)
    imag = -torch.tensor([d[3] for d in data]).div(td_scale).view(td_shape)
    return imag, real


@app.cell
def _(plt, real):
    plt.plot(real[0])
    return


@app.cell
def _(algo, imag, real, sched, torch):
    y = torch.complex(real, imag).t().clone().cuda()
    ids = torch.tensor(sched).cuda()

    n = 2048
    niter = 100
    sigma = 0.05
    tau = 1 / sigma**2

    (mu, gamma, yhat, sigma) = algo.vrls(y, ids, n=n, tau=tau, xi=tau, niter=niter)
    mu = mu.roll(n // 2, dims=1)
    gamma = gamma.roll(n // 2, dims=1)

    f1_hz = 2349.111 + 7002.801 * torch.linspace(-0.5, 0.5, mu.size(0))
    f2_hz = 5979.972 + 1824.568 * torch.linspace(-0.5, 0.5, mu.size(1))
    f1_ppm = f1_hz / 500.132
    f2_ppm = f2_hz / 50.684
    return f1_ppm, f2_ppm, gamma, mu


@app.cell
def _(Path, f1_ppm, f2_ppm, gamma, mu, plt):
    i = 830
    f1_i = f1_ppm[i].item()

    mu_i = mu[i].real.cpu()
    gamma_i = gamma[i].sqrt().cpu()

    recon = mu.real.t().cpu()

    _x = f1_ppm[580:980]
    _y = f2_ppm.clone()
    _z = recon[:, 580:980]

    (fig, (left, right)) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    left.contour(
        _x, _y, _z,
        levels=[0.05, 0.1, 0.2, 0.5, 1],
        cmap="binary",
    )
    left.invert_xaxis()
    left.invert_yaxis()
    left.set_xlabel("$^1$H / ppm")
    left.set_ylabel("$^{15}$N / ppm")
    left.grid(color=(0.9,) * 3)
    left.text(
        10.8, 104, "(a)",
        fontweight="bold",
        horizontalalignment="center",
    )

    right.plot(f2_ppm, mu_i, color=(0.4,) * 3)
    right.plot(f2_ppm, gamma_i, color=(0.6,) * 3, linewidth=1)
    right.set_xlim((108, 135))
    right.set_xlabel(f"$^{{15}}$N / ppm ($f_1 = {f1_i:.2f}$ ppm)")
    right.set_ylabel("Amplitude / a.u.")
    right.grid(color=(0.9,) * 3)
    right.text(
        110, 3, "(b)",
        fontweight="bold",
        horizontalalignment="center",
    )

    inset = right.inset_axes((0.55, 0.5, 0.4, 0.4))
    inset.plot(f2_ppm, mu_i, color=(0.4,) * 3)
    inset.plot(f2_ppm, gamma_i, color=(0.6,) * 3, linewidth=1)
    inset.set_xlim((100, 135))
    inset.set_ylim((-0.05, 0.1))
    inset.grid(color=(0.9,) * 3)

    plt.savefig(
        Path.cwd() / "figure-1.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    fig
    return


if __name__ == "__main__":
    app.run()
