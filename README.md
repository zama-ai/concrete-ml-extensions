# Concrete ML x TFHE-rs

This repository centralizes code written with TFHE-rs used in Concrete ML.

This repository leverages `pyo3` to interface TFHE-rs code in python.

## Prerequisites

Setup the virtual environment. Install the build tool `maturin` and the rust compiler needed. 

```
make install_rs_build_toolchain
poetry install
pip install maturin
```

## How to build

Using maturin in the virtual environment, build the wheel and install it to the virtual
environment. Build the wheel in release mode so that tfhe-rs is built in release as well.

```
source .venv/bin/activate
maturin develop --release
```

## Test

```
make pytest
```

## Usage

```{python}
import concrete_ml_extensions
```


