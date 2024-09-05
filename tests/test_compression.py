import pytest
import concrete_ml_extensions as deai
import numpy as np
import json

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
    
    params_dict = json.loads(deai.default_params());
    inner_size_rounded_input_poly = round_up(inner_size, params_dict["polynomial_size"])
    inner_size_rounded_output_poly = round_up(inner_size, params_dict["packing_ks_polynomial_size"])

    reference = np.dot(a,b)

    n_bits_compute = int(np.log2(reference)) + 1
    params_dict["bits_reserved_for_computation"] = n_bits_compute
    modified_crypto_params = deai.MatmulCryptoParameters.deserialize(json.dumps(params_dict))

    pkey, ckey = deai.create_private_key(modified_crypto_params)
    ciphertext_a = deai.encrypt(pkey, modified_crypto_params, a)

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
    assert inner_size <= params_dict["packing_ks_polynomial_size"]
    assert len(serialized_encrypted_result) / inner_size_rounded_output_poly < 7


