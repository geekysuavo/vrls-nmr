
# Variationally Reweighted Least Squares

This repository is a proving ground for the _variationally reweighted
least squares_ (VRLS) algorithm, in the specialized context of _nonuniformly
sampled nuclear magnetic resonance_ (NUS NMR) acquisition.

## Building

To build the **vrls-nmr** extension module, you will need PyTorch
and a complete compiler toolchain,
```bash
pip install torch pybind11
```
and then build and install the module:
```bash
python3 setup.py build
python3 setup.py bdist_wheel
pip install dist/*whl
```
Running `make` or `make build` will run the same steps.

## Testing

A small set of unit tests is available in the [test](test/) directory
and may be executed using [pytest](https://docs.pytest.org/):
```bash
pip install pytest
pytest test
```

## Notebooks

Analyses in the [notebooks](notebooks/) directory may be opened
using [marimo](https://marimo.io/):
```bash
pip install marimo
cd notebooks/
marimo edit param-tuning.py
```

## Licensing

The **vrls-nmr** extension module is released under the
[MIT license](https://opensource.org/licenses/MIT). See the
[LICENSE.md](LICENSE.md) file for the complete license terms.

