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

You can also use `concrete_ml_extensions` in iOS projects. To do so:

1. Compile library for both iOS and iOS simulator targets (produces two `.a` static libs)
2. Generate Swift bindings (produces `.h`, `.modulemap` and `.swift` wrapper)
3. Package everything into an `.xcframework` (produces `.xcframework`)
4. Use the `.xcframework` in your iOS project

### 1. Compile library:
```shell
    cargo build --no-default-features --features "use_lib2" --lib --release --target aarch64-apple-ios
    cargo build --no-default-features --features "use_lib2" --lib --release --target aarch64-apple-ios-sim
```

Note: Depending on your config, you may need to install 2 additional target architectures.
For Apple Silicon Macs:
```shell
    rustup target add aarch64-apple-ios aarch64-apple-ios-sim
```

### 2. Generate Swift bindings
```shell
    cargo run \
        --bin uniffi-bindgen \
        --no-default-features \
        --features "uniffi/cli use_lib2" \
        generate --library target/aarch64-apple-ios/release/libconcrete_ml_extensions.dylib \
        --language swift \
        --out-dir GENERATED/
```
(Note: you can generate Python bindings as well by replacing `--language swift` with `--language python`)

Now, 3 files have been generated:
- `concrete_ml_extensionsFFI.h`
- `concrete_ml_extensionsFFI.modulemap`.
- `concrete_ml_extensions.swift`

The 2 `*FFI` files compose the low-level C FFI layer: The C header file (.h) declares the low-level structs and functions for calling into Rust, and the .modulemap exposes them to Swift. We'll create a *first* module with these (called concrete_ml_extensions) 

This is enough to call the Rust library from Swift, but in a very verbose way. Instead, you want to use a higher-level Swift API, using the `*.swift` wrapper. This wrapper is uncompiled swift source code; to use it you can:
- Either drag and drop it as source in your app codebase (simpler)
- Or compile it in a second module of its own, and use it as compiled code (more complex)
You can read more about why UniFFI split things that way: https://mozilla.github.io/uniffi-rs/0.27/swift/overview.html

Next steps:
1. Move .h and .module in an include folder, and rename `<name>.modulemap` to `module.modulemap` (.xcframework and Xcode expect this name).
```shell
    mkdir -p GENERATED/include
    mv GENERATED/concrete_ml_extensionsFFI.modulemap GENERATED/include/module.modulemap
    mv GENERATED/concrete_ml_extensionsFFI.h GENERATED/include/concrete_ml_extensionsFFI.h
```

### 3. Package everything (.h, .module, and 2 .a) into an `.xcframework`

```shell
    xcodebuild -create-xcframework \
        -library target/aarch64-apple-ios/release/libconcrete_ml_extensions.a \
        -headers GENERATED/include/ \
        -library target/aarch64-apple-ios-sim/release/libconcrete_ml_extensions.a \
        -headers GENERATED/include/ \
        -output GENERATED/ConcreteMLExtensions.xcframework
```

The generated `.xcframework` works fine if included in a simple test project.
However, if you use it in a project already containing another .xcframework (for ex, TFHE.xcframework), you'll get this error:

> Multiple commands produce '...Xcode/DerivedData/Workspace-ejeewzlcxbwwtbbihtdvnvgjkysh/Build/Products/Debug/include/module.modulemap'

To fix, a workaround [suggested here](https://github.com/jessegrosjean/module-map-error) is to wrap the .h and .modulemap in a subfolder:

```shell
    mkdir -p GENERATED/ConcreteMLExtensions.xcframework/ios-arm64/Headers/concreteHeaders
    mkdir -p GENERATED/ConcreteMLExtensions.xcframework/ios-arm64-simulator/Headers/concreteHeaders
    mv GENERATED/ConcreteMLExtensions.xcframework/ios-arm64/Headers/concrete_ml_extensionsFFI.h \
        GENERATED/ConcreteMLExtensions.xcframework/ios-arm64/Headers/module.modulemap \
        GENERATED/ConcreteMLExtensions.xcframework/ios-arm64/Headers/concreteHeaders
    mv GENERATED/ConcreteMLExtensions.xcframework/ios-arm64-simulator/Headers/concrete_ml_extensionsFFI.h \
        GENERATED/ConcreteMLExtensions.xcframework/ios-arm64-simulator/Headers/module.modulemap \
        GENERATED/ConcreteMLExtensions.xcframework/ios-arm64-simulator/Headers/concreteHeaders
```


### 4. Use the `.xcframework` in your iOS project
- Copy .xcframework into your project
- Add it to `Target > General > Frameworks, Libraries, and Embedded Content`
- Select `Do Not Embed` (it's a static lib)
- Copy `concrete_ml_extensions.swift` (as source code) in project

### Troubleshooting:
    "failed to get iphoneos SDK path: SDK "iphoneos" cannot be located"
    => Ensure Xcode.app/Settings/Locations/Command Line Tools is set to the right version.


## Fast Edit Loop
```shell
    cargo build --no-default-features --features "use_lib2" --lib --target aarch64-apple-ios-sim
    
    cargo run \
        --bin uniffi-bindgen \
        --no-default-features \
        --features "uniffi/cli use_lib2" \
        generate --library target/aarch64-apple-ios-sim/debug/libconcrete_ml_extensions.dylib \
        --language swift \
        --out-dir GENERATED/
        
    mv GENERATED/concrete_ml_extensionsFFI.modulemap GENERATED/module.modulemap
```
