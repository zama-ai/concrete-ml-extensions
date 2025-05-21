SHELL:=$(shell /usr/bin/env which bash)
OS:=$(shell uname)
RS_CHECK_TOOLCHAIN:=$(shell cat toolchain.txt | tr -d '\n')
CARGO_RS_CHECK_TOOLCHAIN:=+$(RS_CHECK_TOOLCHAIN)
CPU_COUNT=$(shell ./scripts/cpu_count.sh)
RS_BUILD_TOOLCHAIN:=stable
CARGO_RS_BUILD_TOOLCHAIN:=+$(RS_BUILD_TOOLCHAIN)
CARGO_PROFILE?=release
MIN_RUST_VERSION:=1.81.0
PYTHON_VERSION:=$(shell python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
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

.PHONY: build_wasm # Build the WASM package used in tests and other applications
build_wasm: install_rs_build_toolchain
	@echo "Building WASM package..."
	@mkdir -p rust/pkg-wasm # Ensure the output directory exists
	cd rust && RUSTFLAGS="" PYO3_CROSS_PYTHON_VERSION=$(PYTHON_VERSION) PYO3_CROSS=1 wasm-pack build . \
		--target web \
		--out-name concrete_ml_extensions_wasm \
		--out-dir ./pkg-wasm \
		--release \
		-- --no-default-features --features wasm_bindings
	@echo "WASM package built in rust/pkg-wasm"

.PHONY: test # Run the tests
test: install_rs_check_toolchain
	RUSTFLAGS="$(RUSTFLAGS)" cargo $(CARGO_RS_CHECK_TOOLCHAIN) test --profile $(CARGO_PROFILE)

.PHONY: fmt # Format rust code
fmt: install_rs_check_toolchain
	cargo "$(CARGO_RS_CHECK_TOOLCHAIN)" fmt --manifest-path rust/Cargo.toml
	black .

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

.PHONY: build_fhe_server # Build the FHE server (helper for WASM tests)
build_fhe_server: install_rs_build_toolchain
	@echo "Building FHE server..."
	cd tests/test_wasm/backend && cargo build --release

.PHONY: run_fhe_server # Run the FHE server (helper for WASM tests)
run_fhe_server: build_fhe_server
	@echo "Starting FHE server..."
	cd tests/test_wasm/backend && ./target/release/backend

.PHONY: wasm_build_env # Builds WASM client, FHE server, and copies artifacts to frontend.
wasm_build_env: build_wasm build_fhe_server
	@echo "Copying WASM artifacts to frontend..."
	@mkdir -p tests/test_wasm/frontend/pkg
	@cp -r rust/pkg-wasm/* tests/test_wasm/frontend/pkg/
	@echo "WASM artifacts copied to tests/test_wasm/frontend/pkg"
	@echo "WASM build environment preparation complete"

.PHONY: wasm_clean # Clean all WASM-related build and test artifacts.
wasm_clean:
	@echo "Cleaning WASM test build artifacts..."
	rm -rf rust/pkg-wasm
	rm -rf tests/test_wasm/frontend/pkg
	rm -rf tests/test_wasm/frontend/node_modules
	rm -rf tests/test_wasm/frontend/playwright-report
	rm -rf tests/test_wasm/frontend/test-results
	cd tests/test_wasm/backend && cargo clean

.PHONY: wasm_dev_server # Sets up environment and runs FHE server for manual WASM testing in browser.
wasm_dev_server: wasm_build_env
	@echo "Starting WASM test environment for manual interaction..."
	@echo "Open http://localhost:8000 in your browser"
	@echo "Press Ctrl+C to stop the FHE server"
	@make run_fhe_server

.PHONY: wasm_test_e2e_install_deps # Install Playwright and its browser dependencies for E2E tests
wasm_test_e2e_install_deps:
	@echo "Installing Playwright test dependencies..."
	cd tests/test_wasm/frontend && npm install --quiet
	@echo "Installing Playwright browsers..."
	cd tests/test_wasm/frontend && npx playwright install --with-deps

.PHONY: wasm_test_e2e # Run full WASM E2E tests (cleans, builds env, installs deps, executes tests). For CI & local.
wasm_test_e2e: wasm_clean wasm_build_env wasm_test_e2e_install_deps
	@echo "Starting FHE server for E2E tests..."
	cd tests/test_wasm/backend && cargo run --release & \
	FHE_SERVER_PID=$$!; \
	echo "FHE server started with PID $${FHE_SERVER_PID}"; \
	echo "Waiting for server to be ready..."; \
	sleep 8; \
	echo "Running Playwright E2E tests..."; \
	(cd tests/test_wasm/frontend && npx playwright test e2e.spec.js) ; EXIT_CODE=$$?; \
	echo "Stopping FHE server (PID $${FHE_SERVER_PID})..."; \
	kill $${FHE_SERVER_PID} || echo "Server (PID $${FHE_SERVER_PID}) already stopped or kill failed."; \
	wait $${FHE_SERVER_PID} 2>/dev/null || echo "Server process (PID $${FHE_SERVER_PID}) finished."; \
	echo "E2E tests finished with exit code $${EXIT_CODE}."; \
	exit $${EXIT_CODE}
