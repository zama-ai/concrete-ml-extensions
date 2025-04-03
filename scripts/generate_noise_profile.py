import concrete_ml_extensions as fhext
import numpy as np
import json
import time
import argparse
import os
from tqdm import tqdm
import hashlib
import warnings
import math

def get_crypto_params_hash(params_dict: dict) -> str:
    """Generates a SHA256 hash for the crypto parameters dictionary."""
    serialized_params = json.dumps(params_dict, sort_keys=True)
    return hashlib.sha256(serialized_params.encode()).hexdigest()[:16]

def run_single_fhe_trial(pkey, ckey, crypto_params_obj, params_dict, matrix_a_shape, matrix_b_shape):
    """
    Generates data, scales inputs to approach target bit width (floor scaling),
    runs FHE matmul, decrypts, and returns results.
    """
    # Setup & Data Generation
    n_bits_input = 8
    low = -(2**(n_bits_input - 1))
    high = 2**(n_bits_input - 1)
    a_int64_initial = np.random.randint(low, high, size=matrix_a_shape, dtype=np.int64)
    b_int64 = np.random.randint(low, high, size=matrix_b_shape, dtype=np.int64)

    # Scaling Calculation
    n_bits_compute = params_dict["bits_reserved_for_computation"]
    if n_bits_compute is None or n_bits_compute <= 0:
            raise ValueError("Invalid bits_reserved_for_computation in crypto_params")

    target_max_magnitude = (2**(n_bits_compute - 1)) - 1
    scale_factor_int = 1

    reference_unscaled = a_int64_initial @ b_int64
    max_initial_abs = np.max(np.abs(reference_unscaled)) if reference_unscaled.size > 0 else 0

    if max_initial_abs > 0 and target_max_magnitude > 0 and max_initial_abs < target_max_magnitude :
        scale_factor_int = math.floor(target_max_magnitude / max_initial_abs)
        scale_factor_int = max(1, scale_factor_int) # Ensure at least 1

    # Apply scaling safely
    try:
            a_int64_scaled = (a_int64_initial * scale_factor_int).astype(np.int64)
            
            # Add controlled LSB variation
            # Calculate how many LSBs are likely zeros due to scaling
            lsb_zeros = 0
            temp_scale = scale_factor_int
            while temp_scale > 1 and temp_scale % 2 == 0:
                lsb_zeros += 1
                temp_scale //= 2
            
            if lsb_zeros > 0:
                # Generate random LSB noise within the zero bits
                lsb_noise = np.random.randint(0, 2**lsb_zeros, size=a_int64_scaled.shape, dtype=np.int64)
                # Add the noise (this only affects the zeroed LSBs)
                a_int64_scaled = a_int64_scaled + lsb_noise
                
    except OverflowError:
            print(f"OverflowError during scaling multiplication even after clamping (Scale: {scale_factor_int}). Skipping trial.")
            return None, None

    # Cleartext Reference
    reference = a_int64_scaled @ b_int64

    # Prepare FHE Inputs
    a_uint64 = a_int64_scaled.astype(np.uint64)
    b_uint64 = b_int64.astype(np.uint64)

    # FHE Computation
    encrypted_matrix = fhext.encrypt_matrix(
        pkey=pkey, crypto_params=crypto_params_obj, data=a_uint64
    )

    encrypted_result = fhext.matrix_multiplication(
        encrypted_matrix=encrypted_matrix,
        data=b_uint64,
        compression_key=ckey,
    )

    # FHE Decryption
    polynomial_size = params_dict["polynomial_size"]
    cols_b = matrix_b_shape[1]
    num_valid_glwe_values = cols_b % polynomial_size or polynomial_size

    decrypted_result_uint64 = fhext.decrypt_matrix(
        encrypted_result,
        pkey,
        crypto_params_obj,
        num_valid_glwe_values_in_last_ciphertext=num_valid_glwe_values,
    )
    fhe_result = decrypted_result_uint64.astype(np.int64)

    return reference, fhe_result


def analyze_bit_errors(reference, result, max_bit_position=None):
    """
    Analyze individual bit errors between reference and result matrices.
    Uses vectorized operations and robust bit width calculation.

    Args:
        reference: Reference matrix from clear computation (np.int64)
        result: Matrix from homomorphic computation or noise model (np.int64)
        max_bit_position: Maximum bit position to analyze (e.g., n_bits_compute).
                          If None, determined automatically based on data range.

    Returns:
        bit_error_rates: Percentage of elements with errors at each bit position (LSB first)
                         Array length is max_bit_position.
    """
    # Ensure consistent data types (should already be int64 from main loop)
    reference = reference.astype(np.int64)
    result = result.astype(np.int64)

    # XOR matrices to identify differences at bit level
    xor_result = reference ^ result
    total_elements = reference.size
    if total_elements == 0:
        # Handle empty input case - return array of zeros matching expected length
        if max_bit_position is None or max_bit_position <= 0:
             max_bit_position = 1 # Default to 1 if not specified and empty
        return np.zeros(max_bit_position)

    # Determine the number of bits to analyze
    effective_max_bits = max_bit_position
    if effective_max_bits is None:
        # Automatically determine based on data range if not provided
        max_abs_ref = np.max(np.abs(reference))
        max_abs_res = np.max(np.abs(result))
        max_value = max(max_abs_ref, max_abs_res)

        # Calculate bits needed for magnitude + 1 sign bit.
        if max_value == 0:
            effective_max_bits = 1 # Need at least 1 bit for 0
        else:
            # Calculate bits for magnitude: ceil(log2(max_value + 1))
            # Add 1 for the sign bit.
            magnitude_bits = int(np.ceil(np.log2(max_value + 1)))
            effective_max_bits = magnitude_bits + 1

        # Ensure minimum of 1 bit position
        effective_max_bits = max(1, effective_max_bits)

    # Count errors at each bit position using vectorized operations
    bit_error_counts = np.zeros(effective_max_bits, dtype=np.int64)
    flat_xor = xor_result.ravel()

    # Precompute powers of 2 (masks)
    masks = np.int64(1) << np.arange(effective_max_bits, dtype=np.int64)

    for bit_pos in range(effective_max_bits):
        # Count how many elements have this bit set in the XOR result
        errors_at_pos = np.count_nonzero((flat_xor & masks[bit_pos]) != 0)
        bit_error_counts[bit_pos] = errors_at_pos

    # Calculate error rates as percentages
    bit_error_rates = (bit_error_counts / total_elements) * 100

    return bit_error_rates

def update_manifest(manifest_path: str, profile_key: str, profile_data: dict):
    """Reads, updates, and writes the MANIFEST.json file.

    Args:
        manifest_path: Full path to the MANIFEST.json file.
        profile_key: The key for the profile entry (e.g., "a1x2048_b2048x2048_chash_abc123").
        profile_data: The dictionary containing profile metadata (filename, hash, etc.).
    """
    manifest_data = {"default_profile_key": None, "profiles": {}}
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
                if not isinstance(manifest_data, dict):
                     warnings.warn(f"Manifest file '{manifest_path}' does not contain a dictionary. Resetting.")
                     manifest_data = {"default_profile_key": None, "profiles": {}}
                if "profiles" not in manifest_data or not isinstance(manifest_data.get("profiles"), dict):
                     warnings.warn(f"Manifest file '{manifest_path}' missing or invalid 'profiles' dictionary. Resetting profiles.")
                     manifest_data["profiles"] = {}
                if "default_profile_key" not in manifest_data:
                    manifest_data["default_profile_key"] = None # Ensure key exists
        else:
            print(f"Manifest file '{manifest_path}' not found. Creating a new one.")

    except json.JSONDecodeError:
        warnings.warn(f"Could not decode JSON from '{manifest_path}'. File might be corrupt. Creating a new structure.")
        manifest_data = {"default_profile_key": None, "profiles": {}}
    except Exception as e:
        warnings.warn(f"Error reading manifest file '{manifest_path}': {e}. Attempting to overwrite.")
        manifest_data = {"default_profile_key": None, "profiles": {}}

    # Add or update the profile entry
    print(f"Updating manifest with key: '{profile_key}'")
    manifest_data["profiles"][profile_key] = profile_data

    try:
        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2) # Use indent for readability
        print(f"Manifest file '{manifest_path}' updated successfully.")
    except Exception as e:
        print(f"Error: Could not write updated manifest to '{manifest_path}': {e}")
        print("Please check file permissions and disk space.")

def main():
    parser = argparse.ArgumentParser(description="Generate FHE noise profile for matrix multiplication.")
    parser.add_argument(
        "--output-dir", default="src/concrete_ml_extensions/noise_profiles",
        help="Directory to save the generated noise profile JSON file."
    )
    parser.add_argument(
        "-n", "--num-runs", type=int, default=100,
        help="Number of FHE computations to average over."
    )
    parser.add_argument(
        "--inner-dim", type=int, default=2048, help="Inner dimension for matmul (cols A, rows B)."
    )
    parser.add_argument(
        "--rows-a", type=int, default=1, help="Number of rows for matrix A (batch size)."
    )
    parser.add_argument(
        "--cols-b", type=int, default=1024, help="Number of columns for matrix B (output features)."
    )
    parser.add_argument(
        "--crypto-params", default=None,
        help="Path to a JSON file containing crypto parameters (optional, defaults to package default)."
    )
    args = parser.parse_args()

    print("--- Noise Profile Generation ---")
    print(f"Number of runs: {args.num_runs}")
    print(f"Matrix A shape: ({args.rows_a}, {args.inner_dim})")
    print(f"Matrix B shape: ({args.inner_dim}, {args.cols_b})")
    print(f"Inner dimension: {args.inner_dim}")

    # Load or define crypto parameters
    params_source = "default"
    if args.crypto_params:
        print(f"Loading crypto parameters from: {args.crypto_params}")
        try:
            with open(args.crypto_params, 'r') as f:
                params_dict = json.load(f)
            params_source = args.crypto_params
        except Exception as e:
            print(f"Error loading crypto params file: {e}")
            return
    else:
        print("Using default crypto parameters.")
        params_dict = json.loads(fhext.default_params())

    try:
        # Create crypto params object
        crypto_params_obj = fhext.MatmulCryptoParameters.deserialize(json.dumps(params_dict))
        # Ensure params_dict reflects the actual object state after potential internal adjustments
        params_dict = json.loads(crypto_params_obj.serialize())
    except Exception as e:
        print(f"Error processing crypto parameters: {e}")
        return

    # Calculate hash based on the final parameters used
    crypto_hash = get_crypto_params_hash(params_dict)
    print(f"Crypto parameters hash: {crypto_hash}")

    # Generate Keys (once for all runs)
    print("Generating FHE keys...")
    start_keygen = time.time()
    try:
        pkey, ckey = fhext.create_private_key(crypto_params_obj)
        print(f"Key generation took {time.time() - start_keygen:.2f}s")
    except Exception as e:
        print(f"Error during key generation: {e}")
        return

    all_error_rates = []
    max_bits_observed_run = 0
    print(f"Running {args.num_runs} FHE trials...")
    successful_runs = 0
    for _ in tqdm(range(args.num_runs), desc="FHE Trials"):
        ref, fhe_res = run_single_fhe_trial(
            pkey, ckey, crypto_params_obj, params_dict,
            (args.rows_a, args.inner_dim), (args.inner_dim, args.cols_b)
        )
        if ref is None:
            print("Warning: Skipping run due to FHE error.")
            continue

        successful_runs += 1
        max_bits_ref = fhext.calculate_bit_width(ref)
        error_rates_lsb = analyze_bit_errors(ref, fhe_res, max_bit_position=max_bits_ref)
        all_error_rates.append(error_rates_lsb)
        max_bits_observed_run = max(max_bits_observed_run, len(error_rates_lsb))

    if successful_runs == 0:
        print("Error: No successful FHE runs completed. Cannot generate profile.")
        return
    print(f"Completed {successful_runs}/{args.num_runs} FHE trials successfully.")

    # Pad results to the maximum observed bit length across all runs
    padded_error_rates = [
        np.pad(rates, (0, max_bits_observed_run - len(rates)), 'constant', constant_values=0)
        for rates in all_error_rates
    ]
    error_rates_array = np.array(padded_error_rates)

    # Calculate mean and std dev (LSB first)
    mean_errors_lsb = np.mean(error_rates_array, axis=0)
    std_errors_lsb = np.std(error_rates_array, axis=0)

    # Reverse to MSB first for storage (standard way of thinking about bits)
    mean_errors_msb = mean_errors_lsb[::-1]
    std_errors_msb = std_errors_lsb[::-1]

    # Prepare output data
    profile_metadata = {
        "description": f"Experimentally generated noise profile for FHE matmul.",
        "source_crypto_params_file": params_source,
        "crypto_params_hash": crypto_hash,
        "generation_details": {
            "num_runs": successful_runs,
            "inner_dim": args.inner_dim,
            "matrix_a_shape": [args.rows_a, args.inner_dim],
            "matrix_b_shape": [args.inner_dim, args.cols_b],
            "input_n_bits": 8,
            "input_signed": True
        },
        "max_bits_observed_during_generation": max_bits_observed_run,
        "crypto_parameters_used": params_dict
    }
    output_data = {
        "metadata": profile_metadata,
        "distribution": {
            "means_percent_msb_first": mean_errors_msb.tolist(),
            "stds_percent_msb_first": std_errors_msb.tolist()
        }
    }

    # Determine output filename based on inner dimension and crypto hash
    profile_filename = f"profile_inner{args.inner_dim}_chash{crypto_hash}.json"
    output_path = os.path.join(args.output_dir, profile_filename)

    # Save to JSON
    print(f"Saving noise profile to: {output_path}")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print("Profile saved successfully.")
    except Exception as e:
        print(f"Error saving output file: {e}")
        return

    # Automatically Update Manifest
    manifest_path = os.path.join(args.output_dir, "MANIFEST.json")
    manifest_key = f"inner{args.inner_dim}_chash{crypto_hash}"
    manifest_entry_data = {
        "filename": profile_filename,
        "crypto_params_hash": crypto_hash,
        "description": f"Generated profile for inner dimension {args.inner_dim}, params hash {crypto_hash[:6]}...",
        "inner_dim": args.inner_dim,
        "matrix_a_shape": [args.rows_a, args.inner_dim],
        "matrix_b_shape": [args.inner_dim, args.cols_b]
    }

    update_manifest(manifest_path, manifest_key, manifest_entry_data)

if __name__ == "__main__":
    main()