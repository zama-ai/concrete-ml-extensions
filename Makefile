SHELL:=$(shell /usr/bin/env which bash)
OS:=$(shell uname)
RS_CHECK_TOOLCHAIN:=$(shell cat toolchain.txt | tr -d '\n')
CARGO_RS_CHECK_TOOLCHAIN:=+$(RS_CHECK_TOOLCHAIN)
CPU_COUNT=$(shell ./scripts/cpu_count.sh)
RS_BUILD_TOOLCHAIN:=stable
CARGO_RS_BUILD_TOOLCHAIN:=+$(RS_BUILD_TOOLCHAIN)
CARGO_PROFILE?=release
MIN_RUST_VERSION:=1.81.0
export RUSTFLAGS?=-C target-cpu=native


.PHONY: rs_check_toolchain # Echo the rust toolchain used for checks
rs_check_toolchain:
	@echo $(RS_CHECK_TOOLCHAIN)

.PHONY: rs_build_toolchain # Echo the rust toolchain used for builds
rs_build_toolchain:
	@echo $(RS_BUILD_TOOLCHAIN)

.PHONY: install_rs_check_toolchain # Install the toolchain used for checks
install_rs_check_toolchain:
	@rustup toolchain list | grep -q "$(RS_CHECK_TOOLCHAIN)" || \
	rustup toolchain install -y --profile default "$(RS_CHECK_TOOLCHAIN)" || \
	( echo "Unable to install $(RS_CHECK_TOOLCHAIN) toolchain, check your rustup installation. \
	Rustup can be downloaded at https://rustup.rs/" && exit 1 )

.PHONY: install_rs_build_toolchain # Install the toolchain used for builds
install_rs_build_toolchain:
	@( rustup toolchain list | grep -q "$(RS_BUILD_TOOLCHAIN)" && \
	./scripts/check_cargo_min_ver.sh \
	--rust-toolchain "$(CARGO_RS_BUILD_TOOLCHAIN)" \
	--min-rust-version "$(MIN_RUST_VERSION)" ) || \
	rustup toolchain install --profile default "$(RS_BUILD_TOOLCHAIN)" || \
	( echo "Unable to install $(RS_BUILD_TOOLCHAIN) toolchain, check your rustup installation. \
	Rustup can be downloaded at https://rustup.rs/" && exit 1 )

.PHONY: test # Run the tests
test: install_rs_check_toolchain
	RUSTFLAGS="$(RUSTFLAGS)" cargo $(CARGO_RS_CHECK_TOOLCHAIN) test --profile $(CARGO_PROFILE)

.PHONY: fmt # Format rust code
fmt: install_rs_check_toolchain
	cargo "$(CARGO_RS_CHECK_TOOLCHAIN)" fmt --manifest-path rust/Cargo.toml

.PHONY: check_fmt # Check rust code format
check_fmt: install_rs_check_toolchain
	cargo "$(CARGO_RS_CHECK_TOOLCHAIN)" fmt --check

.PHONY: clippy_all_targets # Run clippy lints on all targets (benches, examples, etc.)
clippy_all_targets: install_rs_check_toolchain
	RUSTFLAGS="$(RUSTFLAGS)" cargo "$(CARGO_RS_CHECK_TOOLCHAIN)" clippy --all-targets \
		-- --no-deps -D warnings

.PHONY: pcc # pcc stands for pre commit checks
pcc: check_fmt clippy_all_targets

.PHONY: help # Generate list of targets with descriptions
help:
	@grep '^\.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1\t\2/' | expand -t30 | sort

.PHONY: pytest
pytest:
	poetry run pytest ./tests -svv --capture=tee-sys --cache-clear

.PHONY: build_dev_cpu
build_dev_cpu:
	maturin develop --release

.PHONY: wheel_cpu
wheel_cpu:
	maturin build --release

.PHONY: check_version_is_consistent
check_version_is_consistent:
	poetry run python scripts/version_utils.py check-version --file-vars "rust/Cargo.toml:package.version"
