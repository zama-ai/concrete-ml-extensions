# tests/test_simulation.py

import pytest
import numpy as np
import concrete_ml_extensions as fhext
import json
import time
from tqdm import tqdm

from concrete_ml_extensions.utils_simulation import analyze_bit_errors, calculate_bit_width


@pytest.mark.parametrize("num_test_runs", [10])
@pytest.mark.parametrize("inner_dim_a_cols", [2048, 4096])
@pytest.mark.parametrize("rows_a", [1])
def test_fhe_vs_simulation_accuracy(num_test_runs, rows_a, inner_dim_a_cols):
    """ Compares the statistical distribution FHE and simulation."""
    print(f"\n--- Testing Simulation Accuracy ({num_test_runs} runs) ---")
    print(f"    Inner dimension: {inner_dim_a_cols}")

    matrix_a_shape = (rows_a, inner_dim_a_cols)
    matrix_b_shape = (inner_dim_a_cols, 2048)
    n_bits_input = 8
    low = -(2**(n_bits_input - 1))
    high = 2**(n_bits_input - 1)

    # Set crypto parameters
    crypto_params_str = fhext.default_params()
    crypto_params_dict = json.loads(crypto_params_str)

    # Create the final crypto object
    crypto_params_obj = fhext.MatmulCryptoParameters.deserialize(json.dumps(crypto_params_dict))
    crypto_params_serialized_for_sim = crypto_params_obj.serialize()
    # Reload dict from object's serialization to be sure they match
    crypto_params_dict = json.loads(crypto_params_serialized_for_sim)

    # Get polynomial size for later use
    poly_size = crypto_params_dict["polynomial_size"]

    print("Generating keys for FHE...")
    start_key = time.time()
    pkey, ckey = fhext.create_private_key(crypto_params_obj)
    print(f"Keygen done in {time.time()-start_key:.2f}s")

    all_fhe_error_rates = []
    all_sim_error_rates = []
    max_bits_observed = 0

    print(f"Running {num_test_runs} comparison trials...")
    for i in tqdm(range(num_test_runs), desc="Accuracy Trials", disable=None):
        # Generate new data for each run (use int64 for reference, uint64 for FHE)
        a_int64 = np.random.randint(low, high, size=matrix_a_shape, dtype=np.int64)
        b_int64 = np.random.randint(low, high, size=matrix_b_shape, dtype=np.int64)
        a_uint64 = a_int64.astype(np.uint64)
        b_uint64 = b_int64.astype(np.uint64)

        # Cleartext Reference
        reference = a_int64 @ b_int64
        actual_result_bits = calculate_bit_width(reference)
        max_bits_observed = max(max_bits_observed, actual_result_bits)

        # FHE Computation
        enc_a = fhext.encrypt_matrix(pkey, crypto_params_obj, a_uint64)
        enc_res = fhext.matrix_multiplication(enc_a, b_uint64, ckey)
        num_valid = 2048 % poly_size or poly_size
        dec_res_uint64 = fhext.decrypt_matrix(enc_res, pkey, crypto_params_obj, num_valid)
        fhe_result = dec_res_uint64.astype(np.int64)

        # Simulation
        # Pass the same serialized crypto params used for the FHE part
        sim_result_flat = fhext.matrix_multiplication_simulate(
            a_int64, b_int64, crypto_params_serialized_for_sim
        )
        sim_result = sim_result_flat.reshape(reference.shape)

        # Analyze Errors (LSB first) for the actual bit width
        fhe_errors_lsb = analyze_bit_errors(reference, fhe_result, max_bit_position=actual_result_bits)
        sim_errors_lsb = analyze_bit_errors(reference, sim_result, max_bit_position=actual_result_bits)

        all_fhe_error_rates.append(fhe_errors_lsb)
        all_sim_error_rates.append(sim_errors_lsb)

    # Process Results
    print(f"Maximum actual result bit width observed across trials: {max_bits_observed}")
    # Pad all rates to the max observed length for averaging
    padded_fhe_rates = [np.pad(r, (0, max_bits_observed - len(r)), 'constant') for r in all_fhe_error_rates]
    padded_sim_rates = [np.pad(r, (0, max_bits_observed - len(r)), 'constant') for r in all_sim_error_rates]

    fhe_rates_arr = np.array(padded_fhe_rates)
    sim_rates_arr = np.array(padded_sim_rates)

    # Calculate mean and std error rates (LSB first)
    mean_fhe_errors = np.mean(fhe_rates_arr, axis=0)
    mean_sim_errors = np.mean(sim_rates_arr, axis=0)
    std_fhe_errors = np.std(fhe_rates_arr, axis=0)
    std_sim_errors = np.std(sim_rates_arr, axis=0)

    # Assert Accuracy
    absolute_tolerance = 2.5  # Allow mean error rate difference up to 2.5 percentage points
    relative_tolerance = 0.15 # Allow 15% relative difference

    print("\n--- Comparison Results (Mean Error Rates % LSB first) ---")
    print(f"Max Bits Considered: {max_bits_observed}")
    print("Bit | FHE Mean | Sim Mean | Abs Diff | FHE Std | Sim Std")
    print("----|----------|----------|----------|---------|---------")
    is_close = True
    mismatched_bits = []
    for bit in range(max_bits_observed):
        fhe_e = mean_fhe_errors[bit]
        sim_e = mean_sim_errors[bit]
        diff = abs(fhe_e - sim_e)
        fhe_s = std_fhe_errors[bit]
        sim_s = std_sim_errors[bit]
        print(f"{bit:3d} | {fhe_e:8.4f} | {sim_e:8.4f} | {diff:8.4f} | {fhe_s:7.4f} | {sim_s:7.4f}")
        # Check tolerance using numpy's isclose for combined relative/absolute check
        if not np.isclose(fhe_e, sim_e, rtol=relative_tolerance, atol=absolute_tolerance):
            print(f"    -> Mismatch at bit {bit} exceeds tolerance (atol={absolute_tolerance}, rtol={relative_tolerance})")
            is_close = False
            mismatched_bits.append(bit)

    assert is_close, (
        f"Mean bit error rates between FHE and simulation diverge significantly at bits: {mismatched_bits}. "
        f"Check comparison table above. Tolerance: atol={absolute_tolerance}, rtol={relative_tolerance}. "
        f"Ensure a noise profile exists and was correctly loaded for inner_dim={inner_dim_a_cols} and the used crypto parameters."
    )

    print("\nSimulation accuracy test passed: Mean error rates match FHE within tolerance.")
