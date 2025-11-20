import marimo

__generated_with = "0.17.8"
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
    return Path, algo, math, np, plt, torch


@app.cell
def _(Path):
    ser_file = Path.cwd() / "nhsqc-ubq.ser"
    assert ser_file.is_file()
    return (ser_file,)


@app.cell
def _(np, ser_file, torch):
    n_td = 1024
    data_scale = 1.0e-14
    be_int = np.dtype(int) #.newbyteorder(">")

    ser_bytes = ser_file.open("rb").read()
    data = np.frombuffer(ser_bytes, dtype=be_int) * data_scale
    data = torch.from_numpy(data).roll(-69)

    ser = torch.complex(data[::2], data[1::2]).view(-1, 2, n_td)
    (p_type, n_type) = ser.unbind(dim=1)
    real = p_type + n_type
    imag = p_type - n_type
    return imag, n_td, real


@app.cell
def _(plt, real):
    plt.plot(real[0].real)
    return


@app.cell
def _(imag, math, n_td, ph0, ph1, plt, real, torch):
    f = torch.linspace(0, 1, n_td)
    ph = torch.exp(2 * math.pi * 1j * (ph0.value + f * ph1.value))

    real_ft = ph * torch.fft.fft(real, dim=1, norm="ortho")
    imag_ft = ph * torch.fft.fft(imag, dim=1, norm="ortho")

    (_, ax) = plt.subplots()
    ax.plot(real_ft[0].real)
    #ax.plot(imag_ft[0].real)
    ax.set_ylim((-0.2, 2))
    ax.grid(alpha=0.2)
    ax
    return imag_ft, real_ft


@app.cell
def _(mo, ph0, ph1):
    mo.vstack([ph0, ph1])
    return


@app.cell
def _(mo):
    ph0 = mo.ui.slider(start=-1, stop=1, step=0.001, value=0.178, debounce=False, full_width=True)
    ph1 = mo.ui.slider(start=-5, stop=5, step=0.001, value=-0.413, debounce=False, full_width=True)
    return ph0, ph1


@app.cell
def _(imag_ft, plt, real_ft):
    slice_index = 196
    slice = real_ft[:, slice_index].real + 1j * imag_ft[:, slice_index].real

    plt.plot(slice.real)
    plt.plot(slice.imag)
    return


@app.cell
def _():
    sched = [
        0, 1, 2, 4, 5, 7, 8, 10, 12, 15, 17, 20, 23, 26, 30, 34, 38,
        43, 48, 53, 59, 66, 74, 82, 91, 100, 111, 122, 135, 149, 164,
        181, 199, 219, 240, 264, 289, 317, 347, 380, 415, 453, 494,
        538, 584, 634, 686, 741, 799, 858, 919, 982,
    ]
    return (sched,)


@app.cell
def _(algo, imag_ft, real_ft, sched, torch):
    y = torch.complex(real_ft.real, imag_ft.real).t().clone().cuda()
    ids = torch.tensor(sched).cuda()

    n = 2048
    sigma = 0.05
    tau = 1 / sigma**2

    (mu, gamma, yhat, sigma) = algo.vrls_mf(y, ids, n=n, tau=tau, xi=tau, niter=500)
    mu = mu.roll(n // 2, dims=1)
    gamma = gamma.roll(n // 2, dims=1)
    return (mu,)


@app.cell
def _(mu, plt):
    plt.plot(mu[601].real.t().cpu())
    plt.plot(mu[400].real.t().cpu())
    return


@app.cell
def _(mu, plt):
    recon = mu[580:620].real.t().cpu()

    (_, cax) = plt.subplots()
    cax.contour(recon, levels=[0.125, 0.25, 0.5, 1, 2, 4])
    cax.grid(alpha=0.2)
    cax
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
