import pytest
import concrete_ml_extensions as fhext
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
    a = np.random.randint(low, high, size=(8, inner_size)).astype(np.uint64)
    b = np.random.randint(low, high, size=(inner_size,inner_size)).astype(np.uint64)
        
    params_dict = json.loads(fhext.default_params());
    inner_size_rounded_input_poly = round_up(a.shape[1], params_dict["polynomial_size"]) * a.shape[0]
    inner_size_rounded_output_poly = a.shape[0] * round_up(b.shape[1], params_dict["packing_ks_polynomial_size"])

    reference = a @ b

    max_value = np.max(np.abs(reference))
    n_bits_compute = int(np.ceil(np.log2(max_value + 1)))

    params_dict["bits_reserved_for_computation"] = n_bits_compute
    modified_crypto_params = fhext.MatmulCryptoParameters.deserialize(json.dumps(params_dict))

    pkey, ckey = fhext.create_private_key(modified_crypto_params)
    ciphertext_a = fhext.encrypt_matrix(pkey, modified_crypto_params, a)

    # Compress and serialize, then decompress    
    serialized_ciphertext = ciphertext_a.serialize()
    assert len(serialized_ciphertext) / inner_size_rounded_input_poly < 5.0
    print(f"Input expansion factor: {len(serialized_ciphertext) / inner_size_rounded_input_poly:.2f}")

    ciphertext_a = fhext.EncryptedMatrix.deserialize(serialized_ciphertext)

    # Perform dot product (server side computation)
    encrypted_result = fhext.matrix_multiplication(ciphertext_a, b, ckey)

    # Performs compression and serialization, then deserialize
    serialized_encrypted_result = encrypted_result.serialize()
    
    # only support single GLWE returned for now
    # when matmul is implemented, relax this
    assert inner_size <= params_dict["packing_ks_polynomial_size"]
    assert len(serialized_encrypted_result) / inner_size_rounded_output_poly < 7
    print(f"Output expansion factor: {len(serialized_encrypted_result) / inner_size_rounded_output_poly:.2f}")


