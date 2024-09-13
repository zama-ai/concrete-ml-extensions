import pytest
import concrete_ml_extensions as deai
import numpy as np
import json



@pytest.mark.parametrize("n_bits", [2, 6, 8])
@pytest.mark.parametrize("dims", [1, 2])
@pytest.mark.parametrize("inner_size", [256, 1024, 2048, 4096])
@pytest.mark.parametrize("signed_b", [False, True])
def test_correctness(n_bits, inner_size, dims, signed_b, crypto_params):
    low_b = -2**(n_bits-1) if signed_b else 0 # randint low value is included
    high_b = 2**(n_bits-1) if signed_b else 2**n_bits # randint high value is not included

    high_a = 2**n_bits

    inner_size_a = 1 if dims == 2 else 0
    inner_size_b = inner_size if dims == 2 else None
    # Randint draws from [low, high).

    if dims == 1:
        a = np.random.randint(0, high_a, size=(inner_size,), dtype=np.int64)
        b = np.random.randint(low_b, high_b, size=(inner_size,), dtype=np.int64)
    else:                
        a = np.random.randint(0, high_a, size=(inner_size_a, inner_size), dtype=np.int64)
        b = np.random.randint(low_b, high_b, size=(inner_size,inner_size_b), dtype=np.int64)

    reference = a @ b

    a = a.astype(np.uint64)
    b = b.astype(np.uint64)

    n_bits_compute = 26 #max(20, int(np.log2(np.max(np.abs(reference)))) + 1)
    params = json.loads(crypto_params.serialize())
    params["bits_reserved_for_computation"] = n_bits_compute
    modified_crypto_params = deai.MatmulCryptoParameters.deserialize(json.dumps(params))

    pkey, ckey = deai.create_private_key(modified_crypto_params)

    if dims == 2:
        encrypted_matrix = deai.encrypt_matrix(pkey=pkey, crypto_params=crypto_params, data=a)        
        matmul_result = deai.matrix_multiplication(
            encrypted_matrix=encrypted_matrix, data=b, compression_key=ckey
        )
        num_valid_glwe_values_in_last_ciphertext = inner_size_b # inner_size_b % params["polynomial_size"]
        decrypted_result = deai.decrypt_matrix(
            matmul_result, pkey, crypto_params, num_valid_glwe_values_in_last_ciphertext=num_valid_glwe_values_in_last_ciphertext
        )
        decrypted_result = decrypted_result.reshape(-1,).astype(np.int64)
    else:
        ciphertext_a = deai.encrypt(pkey, modified_crypto_params, a)

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
        decrypted_result = deai.decrypt(encrypted_result, pkey, modified_crypto_params, num_valid_glwe_values_in_last_ciphertext=1)
        decrypted_result = np.asarray(decrypted_result).astype(np.int64)[0]

    sign_bit = 1 if np.any(reference) < 0 else 0
    msbs_to_check = 12 if n_bits_compute > 12 else n_bits_compute
    shift_delta = 2**(n_bits_compute - msbs_to_check) 
    high_bits = decrypted_result // shift_delta
    high_bits_reference = reference.astype(np.int64) // shift_delta

    if dims == 2:
        assert inner_size_a <= 1
        # FIXME: we should not allow 15% of error, change when noise is fixed
        n_allow_err = max(inner_size, params["polynomial_size"]) * 0.15
        diff = np.abs(high_bits_reference - high_bits) #// (2**(n_bits_compute - msbs_to_check))
        assert(np.sum(diff == 0) > inner_size - n_allow_err)
        assert(np.sum(diff) < n_allow_err)
        assert(np.all(diff) <= 1)
    else:
        assert(high_bits_reference == high_bits)
