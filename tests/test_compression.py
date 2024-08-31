import pytest
import concrete_ml_extensions as deai
import numpy as np

# 2.0f64.powi(2) / 2.0f64.powi(crypto_params.ciphertext_modulus_bit_count as i32)
# STDEV_8B_2048 = 1 / (2.0 ** 29)
PARAMS_8B_2048 = """{
        "bits_reserved_for_computation": 12,
        "glwe_encryption_noise_distribution_stdev": 0.000000002,
        "encryption_glwe_dimension": 1,
        "polynomial_size": 2048,
        "ciphertext_modulus_bit_count": 63,
        "input_storage_ciphertext_modulus": 39,
        "packing_ks_level": 2, 
        "packing_ks_base_log": 14,
        "packing_ks_polynomial_size": 2048,              
        "packing_ks_glwe_dimension": 1,       
        "output_storage_ciphertext_modulus": 26
    }"""

@pytest.mark.parametrize("inner_size", [256, 1024, 4096, 8192]) #, 2048]) #, 14336])
def test_compression(inner_size):
    n_bits = 8
    signed = True
    low = -2**(n_bits-1) if signed else 0 # randint low value is included
    high = 2**(n_bits-1) if signed else 2**n_bits # randint high value is not included

    # Randint draws from [low, high).
    a = np.random.randint(low, high, size=(inner_size,)).astype(np.uint64)
    b = np.random.randint(low, high, size=(inner_size,)).astype(np.uint64)
    
    crypto_params = deai.MatmulCryptoParameters.deserialize(PARAMS_8B_2048)
    pkey, ckey = deai.create_private_key(crypto_params)
    ciphertext_a = deai.encrypt(pkey, crypto_params, a)

    # Compress and serialize, then decompress    
    serialized_ciphertext = ciphertext_a.serialize()
    print(len(serialized_ciphertext))    

    ciphertext_a = deai.CipherText.deserialize(serialized_ciphertext)

    # Perform dot product (server side computation)
    encrypted_result = deai.dot_product(ciphertext_a, b, ckey)

    # Performs compression and serialization, then deserialize
    serialized_encrypted_result = encrypted_result.serialize()
    print(len(serialized_encrypted_result))


