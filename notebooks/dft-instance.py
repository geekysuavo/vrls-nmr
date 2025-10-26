import marimo

__generated_with = "0.17.2"
app = marimo.App()


@app.cell
def _():
    import cmath
    import math
    import pickle
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as F
    return SimpleNamespace, cmath, math, pickle, torch


@app.cell
def _():
    (k, m, n) = (10, 50, 100)
    stdev = 0.01
    return k, m, n, stdev


@app.cell
def _(SimpleNamespace, cmath, k, m, math, n, pickle, stdev, torch):
    ids = torch.randperm(n).narrow(dim=0, start=0, length=m).sort(dim=0).values
    B = torch.eye(n).index_select(dim=0, index=ids)

    omega = cmath.exp(-2j * math.pi / n)
    i = torch.arange(n).unsqueeze(dim=0)
    j = torch.arange(n).unsqueeze(dim=1)
    Phi = omega**(i * j) / math.sqrt(n)
    Phi = Phi.t().conj()  # idft

    A = B.cfloat() @ Phi

    x0 = torch.zeros(n)
    x0_ids = torch.randperm(n).narrow(dim=0, start=0, length=k)
    x0[x0_ids] = 1.0
    x0 = x0.cfloat()

    noise = stdev * torch.randn(m, dtype=torch.cfloat)
    y = A @ x0 + noise

    instance = SimpleNamespace(
        k=k, m=m, n=n, stdev=stdev,
        B=B, Phi=Phi, ids=ids,
        A=A, x0=x0, noise=noise, y=y,
    )
    with open("instance.pickle", "wb") as file:
        pickle.dump(instance, file)
    return


if __name__ == "__main__":
    app.run()
