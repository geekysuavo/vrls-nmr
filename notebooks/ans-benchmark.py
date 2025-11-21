import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from functools import partial, reduce
    from itertools import product
    import operator

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch
    import torch.nn.functional as F

    import vrlsnmr.algorithms as algo
    import vrlsnmr.simulators as sim
    return algo, operator, partial, pd, product, reduce, sim, sns, torch


@app.function
def next_power_of_two(n: int) -> int:
    return 2 ** (n - 1).bit_length()


@app.cell
def _(operator, reduce):
    n = 2048
    niter = 100
    decay_rate = 0.001
    num_replicates = 10

    replicate_values = tuple(range(num_replicates))
    sigma_values = (0.05, 0.1, 0.2)
    ncomp_values = (4, 8)
    resolution_values = (1, 2, 10)
    ampl_dof_values = (1, 10, 50)
    sparsity_values = (0.05, 0.1, 0.2)

    space_names = (
        "replicate",
        "sigma",
        "num_components",
        "resolution",
        "amplitude_dof",
        "min_sparsity",
    )

    space = (
        replicate_values,
        sigma_values,
        ncomp_values,
        resolution_values,
        ampl_dof_values,
        sparsity_values,
    )

    space_size = reduce(operator.mul, map(len, space))
    return decay_rate, n, niter, space, space_names


@app.cell
def _(
    algo,
    decay_rate,
    mo,
    n,
    niter,
    operator,
    partial,
    pd,
    product,
    reduce,
    sim,
    space,
    space_names,
    torch,
):
    data = []

    for point in mo.status.progress_bar(
        product(*space),
        total=reduce(operator.mul, map(len, space)),
    ):
        (rep, sigma, num_components, resolution, amplitude_dof, min_sparsity) = point

        spacing = resolution * decay_rate

        ground_truth = sim.Signal.build(
            num_components=num_components,
            frequencies=partial(sim.random_spaced, lower=-0.5, upper=0.5, space=spacing),
            decayrates=partial(sim.random_chisq, dof=20, mean=decay_rate),
            amplitudes=partial(sim.random_chisq, dof=amplitude_dof, mean=1.0),
            phases=partial(sim.random_normal, mean=0.0, stdev=1.0e-6),
        )
        ground_truth.cuda()

        tau = 1 / sigma**2

        n_final = n
        m_final = int((n // 2) * min_sparsity)

        m_initial = next_power_of_two(int(m_final // 8))
        n_initial = m_initial * 4

        (y, ids, mu, gamma_diag, yhat, sigma_diag) = algo.ans(
            model=algo.vrls,
            measure=partial(ground_truth, noise=sigma),
            m_initial=m_initial,
            m_final=m_final,
            n_initial=n_initial,
            n_final=n_final,
            min_sparsity=min_sparsity,
            tau=tau,
            xi=tau,
            niter=niter,
        )

        mu.squeeze_(dim=0)
        n_fd = mu.size(dim=0)
        t = torch.arange(n_fd, device=mu.device)
        y0 = ground_truth(t).select(dim=0, index=0)
        x0 = torch.fft.fft(y0, norm="ortho").roll(n_fd // 2)

        mse = (mu - x0).abs().square().mean()

        entry = dict(zip(space_names, point))
        entry["m_initial"] = m_initial
        entry["m_final"] = m_final
        entry["n_initial"] = n_initial
        entry["n_final"] = n_final
        entry["mse"] = mse.item()
        data.append(entry)

    df = pd.DataFrame(data)
    df
    return df, ids


@app.cell
def _(ids):
    ids
    return


@app.cell
def _(df, sns):
    sns.histplot(x="mse", hue="sigma", data=df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
