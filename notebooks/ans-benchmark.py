import marimo

__generated_with = "0.19.7"
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

    import pandas as pd
    import torch
    from torch import Tensor
    import torch.nn.functional as F

    import vrlsnmr.algorithms as algo
    import vrlsnmr.operators as op
    import vrlsnmr.simulators as sim
    return Tensor, algo, op, operator, partial, pd, product, reduce, sim, torch


@app.function
def next_power_of_two(n: int) -> int:
    return 2 ** (n - 1).bit_length()


@app.cell
def _(operator, reduce):
    n = 2048
    niter = 100
    decay_rate = 0.001

    ist_mu = 0.98

    sigma_values = (0.01, 0.05, 0.1)
    ncomp_values = (4, 8, 12)
    ampl_dof_values = (1, 10, 50,)
    sparsity_values = (  # sparsity-dependent num_replicas
        (0.05,) * 100 +
        (0.1,) * 25 +
        (0.2,) * 5
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
    return decay_rate, ist_mu, n, niter, space, space_names


@app.cell
def _(partial, sim):
    def build_signal(
        sigma: float,
        num_components: int,
        amplitude_dof: int,
        decay_rate: float,
    ):
        ground_truth = sim.Signal.build(
            num_components=num_components,
            frequencies=partial(
                sim.random_spaced, lower=-0.5, upper=0.5, space=2 * decay_rate
            ),
            decayrates=partial(sim.random_chisq, dof=20, mean=decay_rate),
            amplitudes=partial(sim.random_chisq, dof=amplitude_dof, mean=1.0),
            phases=partial(sim.random_normal, mean=0.0, stdev=1.0e-6),
        )
        ground_truth.cuda()
        measurement = partial(ground_truth, noise=sigma)

        return (ground_truth, measurement)
    return (build_signal,)


@app.cell
def _(Tensor, space_names, torch):
    def append_entries(
        point: tuple,
        x0: Tensor,
        xhat: Tensor,
        mask: Tensor,
        m_final: int,
        n_final: int,
        schedule: str,
        algorithm: str,
        append_to: list[dict],
        var: Tensor | None = None,
        m_initial: int | None = None,
        n_initial: int | None = None,
    ):
        identifier = hash(point)
        replicate = sum(
            entry["identifier"] == identifier
            for entry in append_to
        )
        entry_base = {
            **dict(zip(space_names, point)),
            **dict(
                replicate=replicate,
                identifier=identifier,
                m_initial=m_initial or m_final,
                n_initial=n_initial or n_final,
                m_final=m_final,
                n_final=n_final,
                n_out=x0.size(dim=0),
                schedule=schedule,
                algorithm=algorithm,
            ),
        }

        signal_sumsq = (xhat - x0)[mask].abs().square().sum().item()
        noise_sumsq = (xhat - x0)[~mask].abs().square().sum().item()
        total_sumsq = (xhat - x0).abs().square().sum().item()

        signal_norm = x0[mask].abs().square().sum().item()
        noise_norm = x0[~mask].abs().square().sum().item()
        total_norm = x0.abs().square().sum().item()

        if var is None:
            signal_var = torch.nan
            noise_var = torch.nan
            total_var = torch.nan
        else:
            signal_var = var[mask].mean().item()
            noise_var = var[~mask].mean().item()
            total_var = var.mean().item()

        entry = dict(
            region="signal",
            norm=signal_norm,
            sumsq=signal_sumsq,
            variance=signal_var,
        )
        append_to.append({**entry_base, **entry})

        entry = dict(
            region="noise",
            norm=noise_norm,
            sumsq=noise_sumsq,
            variance=noise_var,
        )
        append_to.append({**entry_base, **entry})

        entry = dict(
            region="total",
            norm=total_norm,
            sumsq=total_sumsq,
            variance=total_var,
        )
        append_to.append({**entry_base, **entry})
    return (append_entries,)


@app.cell
def _(
    algo,
    append_entries,
    build_signal,
    decay_rate,
    ist_mu,
    mo,
    n,
    niter,
    op,
    operator,
    partial,
    product,
    reduce,
    space,
    torch,
):
    data = []

    for point in mo.status.progress_bar(
        product(*space),
        total=reduce(operator.mul, map(len, space)),
    ):
        # === Ground truth ===
        (sigma, num_components, amplitude_dof, min_sparsity) = point
        (ground_truth, measurement) = build_signal(sigma, num_components, amplitude_dof, decay_rate)

        # === Shared params ===
        tau = 1 / sigma**2

        n_final = n
        m_final = int((n // 2) * min_sparsity)

        m_initial = next_power_of_two(int(m_final // 8))
        n_initial = m_initial * 4

        # === ANS + VRLS ===
        (y, ids, mu, gamma_diag, _, _) = algo.ans(
            model=algo.vrls,
            measure=measurement,
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
        gamma_diag.squeeze_(dim=0)

        n_fd = mu.size(dim=0)
        t = torch.arange(n_fd, device=mu.device)
        y0 = ground_truth(t).select(dim=0, index=0)
        x0 = torch.fft.fft(y0, norm="ortho").roll(n_fd // 2)
        x0.real -= x0.real.median()
        mask = x0.real.gt(sigma)

        append = partial(
            append_entries,
            append_to=data,
            point=point,
            x0=x0,
            mask=mask,
            m_initial=m_initial,
            n_initial=n_initial,
            m_final=m_final,
            n_final=n_final,
        )

        append(xhat=mu, var=gamma_diag, schedule="ans", algorithm="vrls")

        # === Static schedules and measurements ===
        ids_unif = op.schedunif(
            m=m_final,
            n=n_fd // 2,
            device=ids.device,
        )

        ids_exp = op.schedexp(
            rate=decay_rate,
            m=m_final,
            n=n_fd // 2,
            device=ids.device,
        )

        ids_pg = op.schedpg(
            m=m_final,
            n=n_fd // 2,
            device=ids.device,
        )

        y_unif = measurement(ids_unif)
        y_exp = measurement(ids_exp)
        y_pg = measurement(ids_pg)

        # === Algorithms ===
        def vrls(y_, ids_):
            (xhat, xvar, _, _) = algo.vrls(y_, ids_, tau=tau, xi=tau, n=n_fd, niter=niter)
            xhat = xhat.squeeze(dim=0).roll(n_fd // 2)
            xvar = xvar.squeeze(dim=0).roll(n_fd // 2)
            return (xhat, xvar)

        def fmf(y_, ids_):
            (xhat, xvar, _, _) = algo.vrls_mf(y_, ids_, tau=tau, xi=tau, n=n_fd, niter=niter * 10)
            xhat = xhat.squeeze(dim=0).roll(n_fd // 2)
            xvar = xvar.squeeze(dim=0).roll(n_fd // 2)
            return (xhat, xvar)

        def ists(y_, ids_):
            (xhat, _) = algo.ists(y_, ids_, mu=ist_mu, n=n_fd, niter=niter * 10)
            xhat = xhat.squeeze(dim=0).roll(n_fd // 2)
            return (xhat, None)

        funcs = dict(vrls=vrls, fmf=fmf, ists=ists)
        inputs = dict(
            ans=(y, ids),
            unif=(y_unif, ids_unif),
            exp=(y_exp, ids_exp),
            pg=(y_pg, ids_pg),
        )

        def run_and_append(schedule: str, algorithm: str):
            (xhat, xvar) = funcs[algorithm](*inputs[schedule])
            append(xhat=xhat, var=xvar, schedule=schedule, algorithm=algorithm)

        for schedule in ("ans", "unif", "exp", "pg"):
            for algorithm in ("vrls", "fmf", "ists"):
                if (schedule, algorithm) != ("ans", "vrls"):
                    run_and_append(schedule, algorithm)
    return (data,)


@app.cell
def _(data, pd):
    df = pd.DataFrame(data)
    df
    return (df,)


@app.cell
def _(df):
    df.to_parquet("ans-benchmark.parquet")
    return


if __name__ == "__main__":
    app.run()
