import numpy as np
import json
import importlib.resources
from typing import Optional, Dict, Any, Tuple
import hashlib
import warnings

def get_crypto_params_hash(params_dict: dict) -> str:
    """Generates a SHA256 hash for the crypto parameters dictionary."""
    serialized_params = json.dumps(params_dict, sort_keys=True)
    return hashlib.sha256(serialized_params.encode()).hexdigest()[:16]

NOISE_PROFILE_DIR_NAME = 'noise_profiles'

def _get_package_resource_path(filename: str):
    """Gets the path to a resource within the noise_profiles directory."""
    with importlib.resources.path('concrete_ml_extensions', NOISE_PROFILE_DIR_NAME) as base_path:
        return base_path / filename

def _load_manifest() -> Dict[str, Any]:
    """Loads the noise profile manifest."""
    manifest_path = _get_package_resource_path('MANIFEST.json')
    if not manifest_path.is_file():
         raise FileNotFoundError(f"Noise profile manifest 'MANIFEST.json' not found at expected location: {manifest_path}")
    try:
        with manifest_path.open('rt', encoding='utf-8') as f:
            manifest_data = json.load(f)
        return manifest_data
    except json.JSONDecodeError as e:
         raise RuntimeError(f"Error decoding noise profile manifest '{manifest_path}': {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading noise profile manifest '{manifest_path}': {e}")

def _get_default_params_hash() -> Optional[str]:
    """Gets the hash of the default crypto parameters."""
    # Need to import late to avoid circular dependency during package init
    from . import concrete_ml_extensions as fhext
    try:
        default_params_serialized = fhext.default_params()
        default_params_dict = json.loads(default_params_serialized)
        return get_crypto_params_hash(default_params_dict)
    except Exception as e:
        warnings.warn(f"Could not determine default crypto parameters hash: {e}")
        return None # Cannot determine default hash

def find_noise_profile(
    matrix_a_shape: Tuple[int, ...],
    matrix_b_shape: Tuple[int, ...],
    crypto_params_serialized: Optional[str] = None
) -> Tuple[str, str]:
    """
    Finds the exact matching noise profile filename and its key based on inner dimension and crypto parameters.

    Args:
        matrix_a_shape: Shape of the first matrix.
        matrix_b_shape: Shape of the second matrix.
        crypto_params_serialized: Serialized JSON string of the crypto parameters to use.
                                  If None, uses default parameters.

    Returns:
        Tuple[str, str]: (profile_key, profile_filename)
    """
    if len(matrix_a_shape) != 2 or len(matrix_b_shape) != 2:
        raise ValueError("Simulation currently only supports 2D matrix multiplication.")
    if matrix_a_shape[1] != matrix_b_shape[0]:
         raise ValueError(f"Incompatible inner dimensions: {matrix_a_shape[1]} != {matrix_b_shape[0]}")

    # Extract inner dimension (cols of A, rows of B)
    inner_dim = matrix_a_shape[1]

    # Need to import late to avoid circular dependency
    from . import concrete_ml_extensions as fhext

    if crypto_params_serialized is None:
        # Use default parameters if none are provided
        crypto_params_serialized = fhext.default_params()
        target_hash = _get_default_params_hash()
        if target_hash is None:
             raise RuntimeError("Could not get hash for default crypto parameters.")
    else:
        try:
            params_dict = json.loads(crypto_params_serialized)
            target_hash = get_crypto_params_hash(params_dict)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for crypto_params_serialized.")

    manifest = _load_manifest()
    profiles = manifest.get("profiles", {})

    # Search for an exact match (Inner Dimension + Crypto Hash)
    for key, profile_info in profiles.items():
        if (profile_info.get("inner_dim") == inner_dim and
            profile_info.get("crypto_params_hash") == target_hash):
            if "filename" not in profile_info:
                 raise KeyError(f"Found matching profile '{key}' but it lacks a 'filename' entry.")
            return key, profile_info["filename"]

    raise KeyError(
        f"No exact noise profile found in manifest for inner_dim={inner_dim}, "
        f"crypto_params_hash='{target_hash}'. "
        f"Consider generating a profile for these parameters."
    )

def load_noise_distribution_from_package(profile_filename: str) -> Dict[str, Any]:
    """Loads a specific noise profile JSON from the package data."""
    profile_path = _get_package_resource_path(profile_filename)
    if not profile_path.is_file():
        raise FileNotFoundError(f"Noise profile '{profile_filename}' not found at expected location: {profile_path}")

    try:
        with profile_path.open('rt', encoding='utf-8') as f:
            noise_data = json.load(f)

        # Validation
        if not isinstance(noise_data, dict) or "distribution" not in noise_data or "metadata" not in noise_data:
             raise KeyError("Noise profile JSON must be a dictionary with 'metadata' and 'distribution' keys.")
        dist = noise_data["distribution"]
        # Updated keys based on generator script
        if not isinstance(dist, dict) or "means_percent_msb_first" not in dist or "stds_percent_msb_first" not in dist:
            raise KeyError("Profile 'distribution' must contain 'means_percent_msb_first' and 'stds_percent_msb_first'.")
        if len(dist["means_percent_msb_first"]) != len(dist["stds_percent_msb_first"]):
            raise ValueError("Length mismatch between 'means_percent_msb_first' and 'stds_percent_msb_first'.")

        # Convert lists back to numpy arrays for easier use internally
        noise_data["distribution"]["means_percent_msb_first"] = np.array(dist["means_percent_msb_first"])
        noise_data["distribution"]["stds_percent_msb_first"] = np.array(dist["stds_percent_msb_first"])

        return noise_data

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from {profile_filename}: {e.msg}", e.doc, e.pos)
    except Exception as e:
        raise RuntimeError(f"Error loading noise profile {profile_filename} from {profile_path}: {e}")


def calculate_bit_width(array: np.ndarray) -> int:
    """Calculate maximum number of bits of a signed integer array."""
    assert array.size > 0
    max_abs_val = np.max(np.abs(array))
    magnitude_bits = int(np.ceil(np.log2(max_abs_val + 1))) if max_abs_val > 0 else 0
    assert magnitude_bits > 0
    return magnitude_bits

def apply_noise_batch(batch: np.ndarray, noise_distribution_data: dict, n_bits_actual_result: int) -> np.ndarray:
    """
    Apply bit-level noise to a batch of integers based on a loaded noise distribution.

    Args:
        batch: A 1D array of integers (int64).
        noise_distribution_data: The loaded noise distribution dictionary (output of load_noise_distribution_from_package).
        n_bits_actual_result: Actual number of bits required for the clean result (determines noise application).

    Returns:
        The batch with noise applied to each bit (int64).
    """
    # Use MSB-first arrays directly from the loaded data
    means_percent_msb = noise_distribution_data["distribution"]["means_percent_msb_first"]
    stds_percent_msb = noise_distribution_data["distribution"]["stds_percent_msb_first"]

    profile_bits = len(means_percent_msb)
    # Use actual result bits to determine how much of the profile noise matters
    bits_to_apply_noise = min(n_bits_actual_result, profile_bits)

    if bits_to_apply_noise <= 0:
        return batch # No bits to apply noise to

    # We need the LSB-first view for application logic below.
    # Take the relevant *last* `bits_to_apply_noise` from the MSB-first profile, then reverse.
    means_to_use_lsb = means_percent_msb[-bits_to_apply_noise:][::-1]
    stds_to_use_lsb = stds_percent_msb[-bits_to_apply_noise:][::-1]

    # Convert percentages to probabilities [0, 1]
    means_prob = means_to_use_lsb / 100.0
    stds_prob = stds_to_use_lsb / 100.0

    batch_size = batch.shape[0]
    assert batch_size > 0

    # Generate random flip probabilities for each (batch_element, bit_position)
    # Sample from a normal distribution centered at the mean error rate for that bit (LSB first)
    flip_probs = np.random.normal(
        loc=means_prob,
        scale=stds_prob,
        size=(batch_size, bits_to_apply_noise)
    )

    # Clip probabilities to be within [0.0, 1.0]
    np.clip(flip_probs, 0.0, 1.0, out=flip_probs)

    # Determine which bits to flip based on the generated probabilities
    flips = (np.random.rand(batch_size, bits_to_apply_noise) < flip_probs)

    # Create bit masks (1, 2, 4, 8, ...) for the bits we are considering (LSB first)
    bit_masks = (1 << np.arange(bits_to_apply_noise, dtype=np.int64))

    # Combine the boolean flips and bit masks to create integer flip masks for each batch element
    flip_masks = flips.astype(np.int64) @ bit_masks

    # Apply the flips to the original batch using XOR
    return batch ^ flip_masks

def simulate_matmul_with_noise(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    crypto_params_serialized = None,
) -> np.ndarray:
    """
    Simulates FHE matrix multiplication by performing cleartext matmul
    and applying noise based on a noise profile matching the dimensions and crypto parameters.

    Args:
        matrix_a: First input matrix (int64 recommended for simulation input).
        matrix_b: Second input matrix (int64 recommended for simulation input).
        crypto_params_serialized: Serialized JSON string of the crypto parameters context.
                                  If None, uses default parameters.

    Returns:
        result_with_noise: Result matrix with noise applied (int64).
    """
    # Perform cleartext matrix multiplication
    reference_result = matrix_a.astype(np.int64) @ matrix_b.astype(np.int64)
    original_shape = reference_result.shape

    # Determine the number of bits required for the actual cleartext result
    n_bits_actual_result = calculate_bit_width(reference_result)

    # Load noise distribution
    try:
            _, profile_filename = find_noise_profile(
                matrix_a.shape, matrix_b.shape, crypto_params_serialized
            )
            noise_distribution_data = load_noise_distribution_from_package(profile_filename)
    except (KeyError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to find or load a suitable noise profile: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during noise profile loading: {e}")

    # Apply noise to the flattened result
    result_flat = reference_result.flatten()
    result_with_noise_flat = apply_noise_batch(
        result_flat, noise_distribution_data, n_bits_actual_result
    )

    # Reshape back to original shape
    result_with_noise = result_with_noise_flat.reshape(original_shape)
    return result_with_noise

def analyze_bit_errors(reference: np.ndarray, result: np.ndarray, max_bit_position: Optional[int] = None) -> np.ndarray:
    """Analyze individual bit errors between reference and result matrices.

    Args:
        reference: Reference matrix from clear computation
        result: Matrix from homomorphic computation or noise model
        max_bit_position: Maximum bit position to analyze. If None, determined automatically.

    Returns:
        bit_error_rates: Percentage of elements with errors at each bit position (LSB first)
    """
    # Ensure consistent data types
    reference = reference.astype(np.int64)
    result = result.astype(np.int64)

    # XOR matrices to identify differences at bit level
    xor_result = reference ^ result
    total_elements = reference.size
    if total_elements == 0:
        return np.array([])

    # If max_bit_position is not specified, determine it based on the maximum value
    if max_bit_position is None:
        max_abs_ref = np.max(np.abs(reference)) if reference.size > 0 else 0
        max_abs_res = np.max(np.abs(result)) if result.size > 0 else 0
        max_value = max(max_abs_ref, max_abs_res)
        max_bit_position = int(np.ceil(np.log2(max_value + 1))) + 1 if max_value > 0 else 1

    # Count errors at each bit position
    bit_error_counts = np.zeros(max_bit_position, dtype=np.int64)
    for value in xor_result.flatten():
         for bit_pos in range(max_bit_position):
             if (value >> bit_pos) & 1:
                 bit_error_counts[bit_pos] += 1

    # Calculate error rates as percentages
    return (bit_error_counts / total_elements) * 100