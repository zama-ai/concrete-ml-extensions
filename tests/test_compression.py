import pytest
import concrete_ml_extensions as deai
import numpy as np
import json

PARAMS_8B_2048 = {
        "bits_reserved_for_computation": 20,
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

def round_up(v, m):
    return (v + m - 1) // m * m

@pytest.mark.parametrize("inner_size", [256, 1024, 2048]) #, 2048]) #, 14336])
def test_compression(inner_size):
    n_bits = 8
    signed = True
    low = -2**(n_bits-1) if signed else 0 # randint low value is included
    high = 2**(n_bits-1) if signed else 2**n_bits # randint high value is not included

    # Randint draws from [low, high).
    a = np.random.randint(low, high, size=(inner_size,)).astype(np.uint64)
    b = np.random.randint(low, high, size=(inner_size,)).astype(np.uint64)
    
    inner_size_rounded_input_poly = round_up(inner_size, PARAMS_8B_2048["polynomial_size"])
    inner_size_rounded_output_poly = round_up(inner_size, PARAMS_8B_2048["packing_ks_polynomial_size"])

    reference = np.dot(a,b)

    n_bits_compute = int(np.log2(reference)) + 1
    params = PARAMS_8B_2048
    params["bits_reserved_for_computation"] = n_bits_compute

    crypto_params = deai.MatmulCryptoParameters.deserialize(json.dumps(params))
    pkey, ckey = deai.create_private_key(crypto_params)
    ciphertext_a = deai.encrypt(pkey, crypto_params, a)

    # Compress and serialize, then decompress    
    serialized_ciphertext = ciphertext_a.serialize()
    assert len(serialized_ciphertext) / inner_size_rounded_input_poly < 5.0

    ciphertext_a = deai.CipherText.deserialize(serialized_ciphertext)

    # Perform dot product (server side computation)
    encrypted_result = deai.dot_product(ciphertext_a, b, ckey)

    # Performs compression and serialization, then deserialize
    serialized_encrypted_result = encrypted_result.serialize()
    
    # only support single GLWE returned for now
    # when matmul is implemented, relax this
    assert inner_size <= PARAMS_8B_2048["packing_ks_polynomial_size"]
    assert len(serialized_encrypted_result) / inner_size_rounded_output_poly < 7


