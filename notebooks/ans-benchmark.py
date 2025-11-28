import marimo

__generated_with = "0.18.1"
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
    from pathlib import Path
    import time

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch
    import torch.nn.functional as F

    import vrlsnmr.algorithms as algo
    import vrlsnmr.simulators as sim
    return (
        Path,
        algo,
        operator,
        partial,
        pd,
        plt,
        product,
        reduce,
        sim,
        sns,
        time,
        torch,
    )


@app.function
def next_power_of_two(n: int) -> int:
    return 2 ** (n - 1).bit_length()


@app.cell
def _(operator, reduce):
    n = 2048
    niter = 100
    decay_rate = 0.001

    sigma_values = (0.01, 0.05, 0.1)
    ncomp_values = (4, 8, 12)
    ampl_dof_values = (1, 10, 50,)
    sparsity_values = (  # sparsity-dependent num_replicas
        (0.05,) * 10 +
        (0.1,) * 5 +
        (0.2,) * 2
    )

    space_names = (
        "sigma",
        "num_components",
        "amplitude_dof",
        "min_sparsity",
    )

    space = (
        sigma_values,
        ncomp_values,
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
    time,
    torch,
):
    data = []

    for point in mo.status.progress_bar(
        product(*space),
        total=reduce(operator.mul, map(len, space)),
    ):
        (sigma, num_components, amplitude_dof, min_sparsity) = point

        ground_truth = sim.Signal.build(
            num_components=num_components,
            frequencies=partial(sim.random_spaced, lower=-0.5, upper=0.5, space=2 * decay_rate),
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

        t0 = time.perf_counter()
        (y, ids, mu, gamma_diag, _, _) = algo.ans(
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
        dt = time.perf_counter() - t0

        mu.squeeze_(dim=0)
        gamma_diag.squeeze_(dim=0)
    
        n_fd = mu.size(dim=0)
        t = torch.arange(n_fd, device=mu.device)
        y0 = ground_truth(t).select(dim=0, index=0)
        x0 = torch.fft.fft(y0, norm="ortho").roll(n_fd // 2)
        x0.real -= x0.real.median()
        mask = x0.real.gt(sigma)

        signal_mse = (mu - x0)[mask].abs().square().mean().item()
        noise_mse = (mu - x0)[~mask].abs().square().mean().item()
        total_mse = (mu - x0).abs().square().mean().item()

        signal_var = gamma_diag[mask].mean().item()
        noise_var = gamma_diag[~mask].mean().item()
        total_var = gamma_diag.mean().item()

        entry = dict(zip(space_names, point))
        entry["m_initial"] = m_initial
        entry["m_final"] = m_final
        entry["n_initial"] = n_initial
        entry["n_final"] = n_final
        entry["mse_signal_ans"] = signal_mse
        entry["mse_noise_ans"] = noise_mse
        entry["mse_total_ans"] = total_mse
        entry["var_signal_ans"] = signal_var
        entry["var_noise_ans"] = noise_var
        entry["var_total_ans"] = total_var
        entry["time_ans"] = dt / niter

        ids_unif = (
            torch.randperm(n_fd // 2, device=ids.device)
            .narrow(dim=0, start=0, length=m_final)
        )
        y_unif = ground_truth(ids_unif, noise=sigma)
        t0 = time.perf_counter()
        (mu_unif, gamma_unif, _, _) = algo.vrls(
            y_unif,
            ids_unif,
            tau=tau,
            xi=tau,
            n=n_fd,
            niter=niter,
        )
        dt = time.perf_counter() - t0
    
        mu_unif = mu_unif.squeeze(dim=0).roll(n_fd // 2)
        gamma_unif = gamma_unif.squeeze(dim=0).roll(n_fd // 2)
    
        signal_mse = (mu_unif - x0)[mask].abs().square().mean().item()
        noise_mse = (mu_unif - x0)[~mask].abs().square().mean().item()
        total_mse = (mu_unif - x0).abs().square().mean().item()

        signal_var = gamma_unif[mask].mean().item()
        noise_var = gamma_unif[~mask].mean().item()
        total_var = gamma_unif.mean().item()

        entry["mse_signal_unif"] = signal_mse
        entry["mse_noise_unif"] = noise_mse
        entry["mse_total_unif"] = total_mse
        entry["var_signal_unif"] = signal_var
        entry["var_noise_unif"] = noise_var
        entry["var_total_unif"] = total_var
        entry["time_unif"] = dt
        data.append(entry)

    df = pd.DataFrame(data)
    df
    return df, mask, mu, mu_unif, x0


@app.cell
def _(df):
    df["var_noise_diff"] = (df.var_noise_unif - df.var_noise_ans) / df.var_noise_unif
    df["mse_noise_diff"] = (df.mse_noise_unif - df.mse_noise_ans) / df.mse_noise_unif
    return


@app.cell
def _(df):
    agg = df.groupby(["sigma", "min_sparsity"])[["mse_noise_ans", "mse_noise_unif", "mse_noise_diff"]].mean()
    agg
    return


@app.cell
def _(Path, df, plt, sns):
    (fig, ax) = plt.subplots(figsize=(6, 4))

    sns.barplot(
        x="sigma",
        y="mse_noise_diff",
        hue="min_sparsity",
        data=df,
        ax=ax,
        palette=[
            (0.8,) * 3,
            (0.6,) * 3,
            (0.4,) * 3,
        ]
    )

    ax.set_xlabel(r"Measurement noise ($\sigma$)")
    ax.set_ylabel("Relative error reduction")
    ax.grid(color=(0.9,) * 3)
    ax.set_axisbelow(True)

    legend = ax.get_legend()
    legend.set_title("Undersampling")
    legend.texts[0].set_text("5%")
    legend.texts[1].set_text("10%")
    legend.texts[2].set_text("20%")

    plt.savefig(
        Path.cwd() / "figure-3.pdf",
        format="pdf",
        dpi=600,
        pad_inches=0,
        bbox_inches="tight",
    )

    fig
    return


@app.cell
def _(mask, mu, mu_unif, plt, x0):
    (_, _ax) = plt.subplots()

    _ax.plot(x0.real.cpu())
    _ax.plot(mu.real.cpu())
    _ax.plot(mu_unif.real.cpu())
    _ax.plot(mask.cpu())
    _ax.grid(alpha=0.2)

    _ax.set_xlim((900, 1024))
    #_ax.set_ylim((-1, 1))

    _ax
    return


if __name__ == "__main__":
    app.run()
