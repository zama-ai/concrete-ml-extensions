import pytest
import concrete_ml_extensions as deai
import numpy as np
import json

# 2.0f64.powi(2) / 2.0f64.powi(crypto_params.ciphertext_modulus_bit_count as i32)
# STDEV_8B_2048 = 1 / (2.0 ** 29)
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

@pytest.mark.parametrize("n_bits", [2, 6, 8])
@pytest.mark.parametrize("inner_size", [256, 1024, 2048, 14336])
@pytest.mark.parametrize("signed", [False]) #, True])
def test_correctness(n_bits, inner_size, signed):
    low = -2**(n_bits-1) if signed else 0 # randint low value is included
    high = 2**(n_bits-1) if signed else 2**n_bits # randint high value is not included

    # Randint draws from [low, high).
    if signed:
        a = np.random.randint(low, 0, size=(inner_size,), dtype=np.int64)
        b = np.random.randint(0, high, size=(inner_size,), dtype=np.int64)
    else:
        a = np.random.randint(low, high, size=(inner_size,), dtype=np.int64)
        b = np.random.randint(low, high, size=(inner_size,), dtype=np.int64)

    reference = np.dot(a,b)

    a = a.astype(np.uint64)
    b = b.astype(np.uint64)

    n_bits_compute = int(np.log2(np.abs(reference))) + 1
    params = PARAMS_8B_2048
    params["bits_reserved_for_computation"] = n_bits_compute

    crypto_params = deai.MatmulCryptoParameters.deserialize(json.dumps(PARAMS_8B_2048))
    pkey, ckey = deai.create_private_key(crypto_params)
    #print(a)
    ciphertext_a = deai.encrypt(pkey, crypto_params, a)

    # Compress and serialize, then decompress    
    serialized_ciphertext = ciphertext_a.serialize()
    ciphertext_a = deai.CipherText.deserialize(serialized_ciphertext)

    # Perform dot product (server side computation)
    encrypted_result = deai.dot_product(ciphertext_a, b, ckey) 

    # Performs compression and serialization, then deserialize
    serialized_encrypted_result = encrypted_result.serialize()
    encrypted_result = deai.CompressedResultCipherText.deserialize(
        serialized_encrypted_result
    )

    # Decrypt result to compare to reference
    decrypted_result = deai.decrypt(encrypted_result, pkey, crypto_params)
        
    
    sign_bit = 1 if reference < 0 else 0
    msbs_to_check = 12 if n_bits_compute > 12 else n_bits_compute
    mask = ((2**(n_bits_compute + sign_bit)) - 1) - (2**(n_bits_compute - msbs_to_check) - 1)
    high_bits = decrypted_result & mask
    high_bits_reference = int(reference) & mask

    assert(np.equal(high_bits_reference, high_bits))
#    assert(np.equal(reference, decrypted_result))

