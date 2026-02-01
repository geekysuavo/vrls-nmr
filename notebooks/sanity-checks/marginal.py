import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from sympy import oo, Rational, Symbol
    from sympy import Abs, exp, integrate, log, simplify, sqrt
    return Abs, Rational, Symbol, exp, integrate, oo, simplify


@app.cell
def _(Symbol):
    w = Symbol("w", real=True, positive=True)
    xi = Symbol("xi", real=True, positive=True)

    x = Symbol("x", complex=True)
    return w, x, xi


@app.cell
def _(Abs, Rational, exp, w, x, xi):
    f = w**(-Rational(3, 2)) * exp(-Rational(1, 2) * (w*Abs(x)**2 + xi/w))
    f
    return (f,)


@app.cell
def _(f, integrate, oo, simplify, w):
    Z = simplify(integrate(f, (w, 0, oo)))
    Z
    return


if __name__ == "__main__":
    app.run()
