__name__ = "concrete-ml-extensions"
__author__ = "Zama"
__all__ = ["concrete-ml-extensions"]
__version__ = "0.2.0"

from .concrete_ml_extensions import *
from .utils_simulation import *

import numpy as np
from typing import Tuple


def create_private_key(crypto_params):
    if is_cuda_enabled() and is_cuda_available():
        return cuda_create_private_key(crypto_params)
    return cpu_create_private_key(crypto_params)


def matrix_multiplication(encrypted_matrix, data, compression_key):
    if is_cuda_enabled() and is_cuda_available():
        if not isinstance(data, CudaClearMatrix):
            clear_matrix_gpu = make_cuda_clear_matrix(data, compression_key)
        else:
            clear_matrix_gpu = data

        return cuda_matrix_multiplication(
            encrypted_matrix, clear_matrix_gpu, compression_key
        )
    return cpu_matrix_multiplication(encrypted_matrix, data, compression_key)


def matrix_multiplication_simulate(
    matrix_data: np.ndarray,
    data: np.ndarray,
    crypto_params_serialized = None,
) -> np.ndarray:
    """Simulates FHE matrix multiplication using noise profiles."""
    if not isinstance(matrix_data, np.ndarray) or not isinstance(data, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")
    if matrix_data.ndim != 2 or data.ndim != 2 :
         raise ValueError("Simulation currently only supports 2D matrix multiplication.")
    if matrix_data.shape[1] != data.shape[0]:
        raise ValueError(f"Incompatible matrix dimensions for multiplication: {matrix_data.shape} x {data.shape}")

    # Matrix multiplication with noise
    result_with_noise = simulate_matmul_with_noise(
        matrix_data, data, crypto_params_serialized
    )

    return result_with_noise


def deserialize_compression_key(data):
    if is_cuda_enabled() and is_cuda_available():
        return CudaCompressionKey.deserialize(data)
    return CpuCompressionKey.deserialize(data)


def encrypt_radix(arr: np.ndarray, secret_key: bytes) -> bytes:
    dtype = arr.dtype

    if not arr.ndim == 2 or dtype.type not in (
        np.uint8,
        np.int8,
        np.int16,
        np.uint16,
        np.uint64,
    ):
        raise AssertionError(
            f"Cannot encrypt datatype {str(dtype)} "
            f"to TFHE-rs serialized ciphertext, only 2-d [u]int8, uint64 ndarrays are supported"
        )
    else:
        if dtype.type is np.uint8:
            return encrypt_serialize_u8_radix_2d(arr, secret_key)
        elif dtype.type is np.int8:
            return encrypt_serialize_i8_radix_2d(arr, secret_key)
        elif dtype.type is np.uint64:
            return encrypt_serialize_u64_radix_2d(arr, secret_key)
        elif dtype.type is np.uint16:
            return encrypt_serialize_u16_radix_2d(arr, secret_key)
        elif dtype.type is np.int16:
            return encrypt_serialize_i16_radix_2d(arr, secret_key)


def decrypt_radix(
    blob: bytes,
    shape: Tuple[int, ...],
    bitwidth: int,
    is_signed: bool,
    secret_key: bytes,
) -> np.ndarray:
    orig_shape = np.asarray(shape)
    assert orig_shape.ndim == 1
    squeezed_shape = np.delete(orig_shape, np.where(orig_shape == 1))
    if squeezed_shape.size == 0:
        squeezed_shape = np.asarray([1, 1])
    elif squeezed_shape.size == 1:
        squeezed_shape = np.concatenate(((1,), squeezed_shape))
    assert squeezed_shape.size <= 2, "Decrypt function only supports 1d or 2d arrays"

    if bitwidth == 16:
        if is_signed:
            result = decrypt_serialized_i16_radix_2d(
                blob, squeezed_shape[1], secret_key
            )
        else:
            result = decrypt_serialized_u16_radix_2d(
                blob, squeezed_shape[1], secret_key
            )
    elif bitwidth == 8:
        if is_signed:
            result = decrypt_serialized_i8_radix_2d(blob, squeezed_shape[1], secret_key)
        else:
            result = decrypt_serialized_u8_radix_2d(blob, squeezed_shape[1], secret_key)
    elif bitwidth == 64:
        if is_signed:
            raise AssertionError(
                "Signed 64-bit integers are not currently supported for Radix decryption."
            )
        else:
            result = decrypt_serialized_u64_radix_2d(
                blob, squeezed_shape[1], secret_key
            )
    else:
        raise AssertionError(
            f"Cannot decrypt {'un' if not is_signed else ''}signed datatype of {str(bitwidth)}b "
            f"from TFHE-rs serialized ciphertext, only [u]int[8,16,64] are supported"
        )
    result = np.reshape(result, tuple(orig_shape))
    return result
