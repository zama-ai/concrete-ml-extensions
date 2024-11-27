__name__ = "concrete-ml-extensions"
__author__ = "Zama"
__all__ = ["concrete-ml-extensions"]
__version__ = "0.1.3"

from .concrete_ml_extensions import *

def create_private_key(crypto_params):
    if is_cuda_enabled() and is_cuda_available():
        return cuda_create_private_key(crypto_params)
    return cpu_create_private_key(crypto_params)

def matrix_multiplication(encrypted_matrix, data, compression_key):
    if is_cuda_enabled() and is_cuda_available():
        return cuda_matrix_multiplication(encrypted_matrix, data, compression_key)
    return cpu_matrix_multiplication(encrypted_matrix, data, compression_key)
