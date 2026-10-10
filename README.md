
# Variationally Reweighted Least Squares

This repository is a proving ground for the _variationally reweighted
least squares_ (VRLS) algorithm, in the specialized context of _nonuniformly
sampled nuclear magnetic resonance_ (NUS NMR) acquisition.

## Building

The **vrls-nmr** project is managed by [uv](https://docs.astral.sh/uv/).
To create the environment and build the extension module, run
```bash
uv sync
```
Running `make` or `make build` will run the same steps, and running
`make again` forces a rebuild.

## Testing

A small set of unit tests is available in the [test](test/) directory
and may be executed using [pytest](https://docs.pytest.org/):
```bash
uv run pytest
```
Running `make test` will run the same steps.

## Linting

The python source files are formatted and linted
with [ruff](https://docs.astral.sh/ruff/) and
type-checked with [ty](https://docs.astral.sh/ty/):
```bash
uv run ruff format --check
uv run ruff check
uv run ty check
```
Running `make lint` will run the same steps.

## Notebooks

Analyses in the [notebooks](notebooks/) directory may be opened
using [marimo](https://marimo.io/):
```bash
uv sync --group notebooks
uv run marimo edit notebooks/param-tuning.py
```

## Licensing

The **vrls-nmr** extension module is released under the
[MIT license](https://opensource.org/licenses/MIT). See the
[LICENSE.md](LICENSE.md) file for the complete license terms.

