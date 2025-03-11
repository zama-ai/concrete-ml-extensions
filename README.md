<p align="center">
<!-- product name logo -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/75a78517-d423-4a28-8db3-1f50e7d86925">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/674c368f-8030-4407-985b-417a09e1fe87">
  <img width=600 alt="Zama Concrete ML">
</picture>
</p>

<hr>

<p align="center">
  <a href="https://docs.zama.ai/concrete-ml"> 📒 Documentation</a> | <a href="https://zama.ai/community"> 💛 Community support</a> | <a href="https://github.com/zama-ai/awesome-zama"> 📚 FHE resources by Zama</a>
</p>

<p align="center">
  <a href="https://github.com/zama-ai/concrete-ml-extensions/releases"><img src="https://img.shields.io/github/v/release/zama-ai/concrete-ml-extensions?style=flat-square"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause--Clear-%23ffb243?style=flat-square"></a>
  <a href="https://github.com/zama-ai/bounty-program"><img src="https://img.shields.io/badge/Contribute-Zama%20Bounty%20Program-%23ffd208?style=flat-square"></a>
  <a href="https://slsa.dev"><img alt="SLSA 3" src="https://slsa.dev/images/gh-badge-level3.svg" /></a>
</p>

## About

### What is Concrete ML Extensions

Concrete ML Extensions by [Zama](https://github.com/zama-ai) is a helper package that helps developers build applications based on **Concrete ML**. It implements low-level cryptographic functions using Fully Homomorphic Encryption (FHE). 
<br></br>

### Main features

- **Fast encrypt-clear matrix multiplication**: This library implements a matrix product between encrypted matrices and clear matrices.
- **Python and Swift bindings for matrix multiplication client applications**: This library contains bindings that help developers build client applications on various platforms, including iOS.
- **Encryption/Decryption to [TFHE-rs](https://docs.zama.ai/concrete-ml) ciphertexts**: To provide interoperability with TFHE-rs ciphertexts, Concrete ML Extensions offers encryption and decryption functions that are used in Concrete ML.

*Learn more about Concrete ML Extensions features in the [documentation](https://docs.zama.ai/concrete-ml).*
<br></br>

## Table of Contents

- **[Installation](#installation)**
- **[Resources](#resources)**
  - [Demos](#demos)
  - [Tutorials](#tutorials)
  - [Documentation](#documentation)
- **[Working with Concrete ML Extensions](#working-with-concrete-ml-extensions)**
  - [Citations](#citations)
  - [Contributing](#contributing)
  - [License](#license)
- **[Support](#support)**
  <br></br>


## Installation

Depending on your OS, Concrete ML Extensions may have GPU support.

|                 OS / HW                 | Available  | Has GPU support |
| :-------------------------------------: | :-----------------: | :--------------: |
|                  Linux x86              |         Yes         |       Yes        |
|                 Windows                 |         No         |        N/A        |
|            macOS 11+ (Intel)            |         Yes         |       No        |
| macOS 11+ (Apple Silicon: M1, M2, etc.) |     Yes     |       No        |

>[!Note]
>Concrete ML Extensions only supports Python `3.8`, `3.9`, `3.10`, `3.11` and `3.12`.

### Pip

Concrete ML Extensions is installed automatically when installing Concrete ML. To install manually from PyPi, run the following:

```
pip install concrete-ml-extensions
```

To use the GPU, a CUDA-enabled GPU with support for CUDA >=11.2 should be available on the target machine.

### From Source For Python

This repository leverages `pyo3` to interface TFHE-rs code in python. First, setup the virtual environment. 
Install the build tool `maturin` and the rust compiler. 

```
make install_rs_build_toolchain
poetry install
pip install maturin
```

Next, using `maturin` in the virtual environment, build the wheel and install it to the virtual
environment. Build the wheel in release mode so that tfhe-rs is built in release as well.

```
source .venv/bin/activate
maturin develop --release
```

To compile for GPU, a CUDA-toolkit version >= 11.2 should be installed on the machine, along with 
a compatible `gcc` version (the package compilation is tested with gcc 11.4).  

### From Source for iOS

You can also use Concrete ML Extensions in iOS projects. To do so:

1. Compile the library for both iOS and iOS simulator targets (produces two `.a` static libs).
2. Generate Swift bindings (produces `.h`, `.modulemap` and `.swift` wrapper).
3. Package everything into an `.xcframework` (produces `.xcframework`).
4. Use the `.xcframework` in your iOS project.

##### 1. Compile the library
```shell
    cargo build --no-default-features --features "use_lib2" --lib --release --target aarch64-apple-ios
    cargo build --no-default-features --features "use_lib2" --lib --release --target aarch64-apple-ios-sim
```

You may need to install 2 additional target architectures.

```shell
    rustup target add aarch64-apple-ios aarch64-apple-ios-sim
```

##### 2. Generate Swift bindings
```shell
    cargo run \
        --bin uniffi-bindgen \
        --no-default-features \
        --features "uniffi/cli swift_bindings" \
        generate --library target/aarch64-apple-ios/release/libconcrete_ml_extensions.dylib \
        --language swift \
        --out-dir GENERATED/
```

Now, three files have been generated in the `GENERATED` subdirectory:
- `concrete_ml_extensionsFFI.h`
- `concrete_ml_extensionsFFI.modulemap`.
- `concrete_ml_extensions.swift`

The two `*FFI` files compose the low-level C FFI layer: The C header file (.h) declares the low-level structs and functions for calling into Rust, and the .modulemap exposes them to Swift. We'll create a *first* module with these (called concrete_ml_extensions) 

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

##### 3. Create an `.xcframework` package

```shell
    xcodebuild -create-xcframework \
        -library target/aarch64-apple-ios/release/libconcrete_ml_extensions.a \
        -headers GENERATED/include/ \
        -library target/aarch64-apple-ios-sim/release/libconcrete_ml_extensions.a \
        -headers GENERATED/include/ \
        -output GENERATED/ConcreteMLExtensions.xcframework
```

##### 4. Use the `.xcframework` in your iOS project
- Copy .xcframework into your project
- Add it to `Target > General > Frameworks, Libraries, and Embedded Content`
- Select `Do Not Embed` (it's a static lib)
- Copy `concrete_ml_extensions.swift` (as source code) in project

##### Troubleshooting:
*Error message*:

```
   "failed to get iphoneos SDK path: SDK "iphoneos" cannot be located"
```    
Solution: Ensure Xcode.app/Settings/Locations/Command Line Tools is set to the right version.

*Error message*:

```
   Multiple commands produce '...Xcode/DerivedData/Workspace-ejeewzlcxbwwtbbihtdvnvgjkysh/Build/Products/Debug/include/module.modulemap'
```

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

<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>

> \[!Note\]
> **Zama 5-Question Developer Survey**
>
> We want to hear from you! Take 1 minute to share your thoughts and helping us enhance our documentation and libraries. 👉 **[Click here](https://www.zama.ai/developer-survey)** to participate.

## Resources

### Demos

- [Encrypted LLM fine-tuning](https://github.com/zama-ai/concrete-ml/tree/main/use_case_examples/lora_finetuning): This demo shows
how to fine-tune a LLM using the Low Rank Approximation approach. It leverages the Concrete ML Extensions package to perform the fine-tuning
on encrypted data. 

*If you have built awesome projects using Concrete ML that leverages the Concrete ML Extensions package, please let us know and we will be happy to showcase them here!*
<br></br>

### Tutorials

Coming soon.

*Explore more useful resources in [Awesome Zama repo](https://github.com/zama-ai/awesome-zama)*
<br></br>

### Documentation

Full, comprehensive documentation is available here: [https://docs.zama.ai/concrete-ml](https://docs.zama.ai/concrete-ml).

<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>

## Working with Concrete ML Extensions

### Citations

To cite Concrete ML Extensions in academic papers, please use the following entry:

```text
@Misc{ConcreteML,
  title={Concrete {ML}: a Privacy-Preserving Machine Learning Library using Fully Homomorphic Encryption for Data Scientists},
  author={Zama},
  year={2022},
  note={\url{https://github.com/zama-ai/concrete-ml}},
}
```

### Contributing

To contribute to Concrete ML Extensions, please refer to [this section of the documentation](docs/developer/contributing.md).
<br></br>

### License

This software is distributed under the **BSD-3-Clause-Clear** license. Read [this](LICENSE) for more details.

#### FAQ

**Is Zama’s technology free to use?**

> Zama’s libraries are free to use under the BSD 3-Clause Clear license only for development, research, prototyping, and experimentation purposes. However, for any commercial use of Zama's open source code, companies must purchase Zama’s commercial patent license.
>
> All our work is open source and we strive for full transparency about Zama's IP strategy. To know more about what this means for Zama product users, read about how we monetize our open source products in [this blog post](https://www.zama.ai/post/open-source).

**What do I need to do if I want to use Zama’s technology for commercial purposes?**

> To commercially use Zama’s technology you need to be granted Zama’s patent license. Please contact us at hello@zama.ai for more information.

**Do you file IP on your technology?**

> Yes, all of Zama’s technologies are patented.

**Can you customize a solution for my specific use case?**

> We are open to collaborating and advancing the FHE space with our partners. If you have specific needs, please email us at hello@zama.ai.

<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>

## Support

<a target="_blank" href="https://zama.ai/community-channels">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/zama-ai/concrete-ml/assets/157474013/86502167-4ea4-49e9-a881-0cf97d141818">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/zama-ai/concrete-ml/assets/157474013/3dcf41e2-1c00-471b-be53-2c804879b8cb">
  <img alt="Support">
</picture>
</a>

🌟 If you find this project helpful or interesting, please consider giving it a star on GitHub! Your support helps to grow the community and motivates further development.

<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>
