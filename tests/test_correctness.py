import pytest
import concrete_ml_extensions as deai
import numpy as np
import json

@pytest.mark.parametrize("n_bits", [2, 6, 8])
@pytest.mark.parametrize("inner_size", [256, 1024, 2048, 14336])
@pytest.mark.parametrize("signed", [False]) #, True])
def test_correctness(n_bits, inner_size, signed, crypto_params):
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
    params = json.loads(crypto_params.serialize())
    params["bits_reserved_for_computation"] = n_bits_compute
    modified_crypto_params = deai.MatmulCryptoParameters.deserialize(json.dumps(params))

    pkey, ckey = deai.create_private_key(modified_crypto_params)
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
    decrypted_result = deai.decrypt(encrypted_result, pkey, modified_crypto_params, num_valid_glwe_values_in_last_ciphertext=1)[0]
    
    sign_bit = 1 if reference < 0 else 0
    msbs_to_check = 12 if n_bits_compute > 12 else n_bits_compute
    mask = ((2**(n_bits_compute + sign_bit)) - 1) - (2**(n_bits_compute - msbs_to_check) - 1)
    high_bits = decrypted_result & mask
    high_bits_reference = int(reference) & mask

    assert(np.equal(high_bits_reference, high_bits))
#    assert(np.equal(reference, decrypted_result))

