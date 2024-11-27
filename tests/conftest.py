import pytest
import concrete_ml_extensions as fhext
import numpy as np
import time
import json

PARAMS_8B_2048 = {
    "bits_reserved_for_computation": 27,
    "glwe_encryption_noise_distribution_stdev": 5.293956729894075e-23,
    "encryption_glwe_dimension": 1,
    "polynomial_size": 2048,
    "ciphertext_modulus_bit_count": 64,
    "input_storage_ciphertext_modulus": 39,
    "packing_ks_level": 2, 
    "packing_ks_base_log": 14,
    "packing_ks_polynomial_size": 2048,              
    "packing_ks_glwe_dimension": 1,       
    "output_storage_ciphertext_modulus": 26,
    "pks_noise_distrubution_stdev": 8.095547030480235e-30
}

PARAMS_NONOISE_8B_2048 = {
    "bits_reserved_for_computation": 32,
    "glwe_encryption_noise_distribution_stdev": 0,
    "encryption_glwe_dimension": 1,
    "polynomial_size": 2048,
    "ciphertext_modulus_bit_count": 64,
    "input_storage_ciphertext_modulus": 39,
    "packing_ks_level": 1, 
    "packing_ks_base_log": 39,
    "packing_ks_polynomial_size": 2048,              
    "packing_ks_glwe_dimension": 1,       
    "output_storage_ciphertext_modulus": 39,
    "pks_noise_distrubution_stdev": 0
}

@pytest.fixture
def crypto_params():
    return fhext.MatmulCryptoParameters.deserialize(json.dumps(PARAMS_8B_2048))

@pytest.fixture
def crypto_params_nonoise():
    return fhext.MatmulCryptoParameters.deserialize(json.dumps(PARAMS_NONOISE_8B_2048))

class Timing:
    def __init__(self, message=""):
        self.message = message

    def __enter__(self):
        print(f"Starting {self.message}")
        self.start = time.time()

    def __exit__(self, *args, **kwargs):
        end = time.time()
        print(f"{self.message} done in {end - self.start} seconds")