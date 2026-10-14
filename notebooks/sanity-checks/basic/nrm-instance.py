import marimo

__generated_with = "0.17.2"
app = marimo.App()


@app.cell
def _():
    import pickle
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as F
    return F, SimpleNamespace, pickle, torch


@app.cell
def _():
    (k, m, n) = (10, 50, 100)
    stdev = 0.01
    return k, m, n, stdev


@app.cell
def _(F, SimpleNamespace, k, m, n, pickle, stdev, torch):
    A = F.normalize(torch.randn(m, n), p=2, dim=1)

    x0 = torch.zeros(n)
    x0_ids = torch.randperm(n).narrow(dim=0, start=0, length=k)
    x0[x0_ids] = 1.0

    noise = stdev * torch.randn(m)
    y = A @ x0 + noise

    instance = SimpleNamespace(
        k=k, m=m, n=n, stdev=stdev,
        A=A, x0=x0, noise=noise, y=y,
    )
    with open("instance.pickle", "wb") as file:
        pickle.dump(instance, file)
    return


if __name__ == "__main__":
    app.run()
