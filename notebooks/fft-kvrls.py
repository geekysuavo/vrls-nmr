import marimo

__generated_with = "0.17.2"
app = marimo.App()


@app.cell
def _():
    import math
    import cmath
    import pickle

    import matplotlib.pyplot as plt
    import torch

    import vrlsnmr
    return pickle, plt, torch, vrlsnmr


@app.cell
def _(pickle):
    with open("instance.pickle", "rb") as file:
        instance = pickle.load(file)
    return (instance,)


@app.cell
def _(instance):
    _k, m, n, stdev = (instance.k, instance.m, instance.n, instance.stdev)
    A, x0, noise, y = (instance.A, instance.x0, instance.noise, instance.y)
    B, Phi, ids = (instance.B, instance.Phi, instance.ids)
    return A, B, Phi, ids, m, n, stdev, x0, y


@app.cell
def _(stdev, torch):
    tau = torch.tensor(1 / stdev**2)  # noise is known
    xi = 1e4
    return tau, xi


@app.cell
def _(n, torch):
    mu = torch.zeros(n).cfloat()
    _Gamma = torch.eye(n).cfloat()
    w = torch.ones(n)
    return mu, w


@app.cell
def _(ids, mu, plt, tau, torch, vrlsnmr, w, x0, xi, y):
    for _ in range(10):
        K = vrlsnmr.kernel(w, ids, tau)
        Kinv = torch.linalg.inv(K)
        mu.mul_(0)
        mu[ids] = Kinv @ y
        mu_1 = torch.fft.fft(mu, norm='ortho') / w
        Gamma_diag = vrlsnmr.xmarginal(Kinv, w, ids)
        m2 = mu_1.abs().square() + Gamma_diag
        w_1 = (xi / (m2 + 1e-09)).sqrt()
    plt.plot(mu_1.real)
    plt.plot(x0.real)
    return Kinv, mu_1, w_1


@app.cell
def _(Kinv, Phi, ids, instance, mu_1, n, plt, torch, vrlsnmr, w_1):
    #plt.plot((Phi @ Gamma @ Phi.t().conj()).diag().real);
    t = torch.arange(n)
    yhat = (Phi @ mu_1).imag
    s = vrlsnmr.ymarginal(Kinv, w_1, ids)
    width = 1000.0
    plt.fill_between(t, y1=yhat + width * s, y2=yhat - width * s, alpha=0.2)
    plt.plot(yhat)
    plt.scatter(instance.ids, instance.y.imag)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Scratchpad
    ## Analyzing `kernel()`
    """
    )
    return


@app.cell
def _(A, m, tau, torch, w_1):
    Winv = w_1.reciprocal().diag().cfloat()
    K_1 = torch.eye(m) / tau + A @ Winv @ A.t().conj()
    return K_1, Winv


@app.cell
def _(Phi, Winv):
    C = Phi @ Winv @ Phi.t().conj()
    c0 = C[:,0]
    return C, c0


@app.cell
def _(n, torch, w_1):
    c = torch.fft.fft(w_1.reciprocal()).conj() / n
    return (c,)


@app.cell
def _(c, c0):
    (c0 - c).abs().max(), (c0 - c).real.abs().max(), (c0 - c).imag.abs().max()
    return


@app.cell
def _(C):
    C[5,2], C[2,5]
    return


@app.cell
def _(c, n):
    c[2 + (n - 5)].conj(), c[2 + (n - 5)]
    return


@app.cell
def _(C, c, n, torch):
    for _i in range(n):
        for _j in range(_i, n):
            assert torch.allclose(C[_i, _j], c[_i - _j])
            if _j != _i:
                assert torch.allclose(C[_j, _i], c[_i - _j].conj())
    return


@app.cell
def _(instance):
    ids_1 = instance.ids
    ids_1
    return (ids_1,)


@app.cell
def _(K_1, c, ids_1, m, n, tau, torch):
    for _i in range(m):
        assert torch.allclose(K_1[_i, _i], c[0] + 1 / tau)
        for _j in range(_i + 1, m):
            _k = n + ids_1[_i] - ids_1[_j]
            assert 0 <= _k < n
            assert torch.allclose(K_1[_i, _j], c[_k])
            assert torch.allclose(K_1[_j, _i], c[_k].conj())
    return


@app.cell
def _(ids_1, tau, vrlsnmr, w_1):
    Kcpp = vrlsnmr.kernel(weights=w_1, ids=ids_1, tau=tau)
    return (Kcpp,)


@app.cell
def _(K_1, Kcpp, torch):
    (K_1.shape == Kcpp.shape, K_1.dtype == Kcpp.dtype, torch.allclose(K_1, Kcpp))
    return


@app.cell
def _(K_1, Kcpp, torch):
    torch.testing.assert_close(K_1, Kcpp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analyzing `xmarginal()`""")
    return


@app.cell
def _(A, Kinv, torch, w_1):
    Gamma_diag_1 = w_1.reciprocal() - torch.sum(Kinv @ A / w_1 * A.conj() / w_1, dim=0).real
    return (Gamma_diag_1,)


@app.cell
def _(Kinv, ids_1, vrlsnmr, w_1):
    gamma = vrlsnmr.xmarginal(Kinv, w_1, ids_1)
    return (gamma,)


@app.cell
def _(Gamma_diag_1, gamma, torch):
    torch.testing.assert_close(Gamma_diag_1, gamma)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analyzing `ymarginal()`""")
    return


@app.cell
def _(A, Phi, m, tau, torch, w_1):
    Winv_1 = w_1.reciprocal().diag().cfloat()
    C_1 = Phi @ Winv_1 @ Phi.t().conj()
    K_2 = torch.eye(m) / tau + A @ Winv_1 @ A.t().conj()
    Kinv_1 = torch.linalg.inv(K_2)
    Gamma_inv = w_1.diag() + tau * A.t().conj() @ A
    _Gamma = torch.linalg.inv(Gamma_inv)
    s_1 = (Phi @ _Gamma @ Phi.t().conj()).real.diag()
    return C_1, Kinv_1, s_1


@app.cell
def _(Kinv_1, ids_1, s_1, torch, vrlsnmr, w_1):
    s_cpp = vrlsnmr.ymarginal(Kinv_1, w_1, ids_1)
    torch.testing.assert_close(s_1, s_cpp)
    return


@app.cell
def _(n, torch, w_1):
    c_1 = torch.fft.fft(w_1.reciprocal()).conj() / n
    return (c_1,)


@app.cell
def _(c_1, ids_1):
    beta = lambda k: c_1.roll(_k).index_select(dim=0, index=ids_1)
    return (beta,)


@app.cell
def _(B, C_1, c_1, ids_1, n, torch):
    for _k in range(n):
        torch.testing.assert_close(c_1.roll(_k).index_select(dim=0, index=ids_1), (B.cfloat() @ C_1.narrow(dim=1, start=_k, length=1)).squeeze(dim=1))
    return


@app.cell
def _(Kinv_1, beta, c_1, torch):
    sigma = lambda k: c_1[0].real - torch.einsum('ij,i,j->', Kinv_1, beta(_k).conj(), beta(_k)).real
    return (sigma,)


@app.cell
def _(n, s_1, sigma, torch):
    for _k in range(n):
        torch.testing.assert_close(s_1[_k], sigma(_k))
    return


@app.cell
def _(beta, c_1, ids_1, m, torch):
    for _j in range(m):
        torch.testing.assert_close(beta(0)[_j], c_1[ids_1[_j]])
    return


@app.cell
def _(beta, c_1, ids_1, m, n, torch):
    for _k in range(n):
        for _j in range(m):
            index = (ids_1[_j] - _k) % n
            assert 0 <= index < n
            torch.testing.assert_close(beta(_k)[_j], c_1[index])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
