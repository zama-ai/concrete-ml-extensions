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

## Compile for iOS & Swift
### Compile (dev mode):
```shell
    cargo build --no-default-features --features "use_lib2"
```

### Generate Swift bindings
```shell
    cargo run \
        --bin uniffi-bindgen \
        --no-default-features \
        --features "uniffi/cli use_lib2" \
        generate --library target/debug/libconcrete_ml_extensions.dylib \
        --language swift \
        --out-dir generated_bindings_swift
```

### Generate Python bindings !
```shell
    cargo run \
        --bin uniffi-bindgen \
        --no-default-features \
        --features "uniffi/cli use_lib2" \
        generate --library target/debug/libconcrete_ml_extensions.dylib \
        --language python \
        --out-dir generated_bindings_python
```

### Install 2 additional target architectures (ios & ios-sim for apple silicon macs):
```shell
    rustup target add aarch64-apple-ios aarch64-apple-ios-sim
```

### Compile for both iOS and iOS simulator targets:
```shell
cargo build --release --no-default-features --features "use_lib2" --target aarch64-apple-ios
cargo build --release --no-default-features --features "use_lib2" --target aarch64-apple-ios-sim
```

### Troubleshoot:
    "failed to get iphoneos SDK path: SDK "iphoneos" cannot be located"
    => Ensure Xcode.app/Settings/Locations/Command Line Tools is set to the right version.
