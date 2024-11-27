import pytest
import concrete_ml_extensions as deai
import numpy as np
import json
import time

CRYPTO_DTYPE = np.uint64

@pytest.mark.parametrize("n_bits", [2, 6, 8])
@pytest.mark.parametrize("dims", [2])
@pytest.mark.parametrize("inner_size", [256, 1024, 2048, 4096])
@pytest.mark.parametrize("signed_b", [False, True]) #FIXME: not working for signed yet.
def test_correctness(n_bits, inner_size, dims, signed_b, crypto_params):

    assert dims == 2

    low_b = -(2 ** (n_bits - 1)) if signed_b else 0  # randint low value is included
    high_b = (
        2 ** (n_bits - 1) if signed_b else 2**n_bits
    )  # randint high value is not included

    high_a = 2**n_bits

    inner_size_a = 8 if dims == 2 else None
    inner_size_b = inner_size if dims == 2 else None

    # Signed values must be processed for the weights, so
    # we generate signed int64. This is also used to compute
    # the max bits 
    a = np.random.randint(
        0, high_a, size=(inner_size_a, inner_size), dtype=np.uint64
    )
    b = np.random.randint(
        low_b, high_b, size=(inner_size, inner_size_b), dtype=np.int64
    )

    # These computations used signed weights
    reference = a @ b

    max_value = np.max(np.abs(reference))
    n_bits_compute = int(np.ceil(np.log2(max_value + 1)))

    assert n_bits_compute <= 27, "n_bits_compute exceeds maximum allowed value"

    # Change the encoding to push the inputs and the result
    # as much as possible to the left in the MSBs
    # in order to avoid noise corruption
    params = json.loads(deai.default_params()) #crypto_params.serialize())
    params["bits_reserved_for_computation"] = (
        n_bits_compute + 1
    )  # +1 for sign bit if needed
#    params["packing_ks_level"] = 1
    modified_crypto_params = deai.MatmulCryptoParameters.deserialize(json.dumps(params))

    pkey, ckey = deai.create_private_key(modified_crypto_params)

    # Need to convert to uint64 since this is what is handled
    # by the crypto
    a = a.astype(CRYPTO_DTYPE)
    b = b.astype(CRYPTO_DTYPE)

    encrypted_matrix = deai.encrypt_matrix(
        pkey=pkey, crypto_params=modified_crypto_params, data=a
    )
    start_time = time.time()

    matmul_result = deai.matrix_multiplication(
        encrypted_matrix=encrypted_matrix, data=b, compression_key=ckey
    )

    tot_server_time = time.time() - start_time
    print(f"Server time without serialization {tot_server_time}s")

    polynomial_size = params["polynomial_size"]
    num_valid_glwe_values_in_last_ciphertext = (
        inner_size_b % polynomial_size or polynomial_size
    )

    decrypted_result = deai.decrypt_matrix(
        matmul_result,
        pkey,
        modified_crypto_params,
        num_valid_glwe_values_in_last_ciphertext=num_valid_glwe_values_in_last_ciphertext,
    )
    decrypted_result = decrypted_result.astype(np.int64)

    # Need to check only MSBS
    # since these are those that are guaranteed
    # to be correct by the crypto-parameters
    msbs_to_check = 12 if n_bits_compute > 12 else n_bits_compute
    shift_delta = 2 ** (n_bits_compute - msbs_to_check)
    high_bits = decrypted_result // shift_delta
    high_bits_reference = reference.astype(np.int64) // shift_delta

    n_allow_err = inner_size * 0.01
    diff = np.abs(
        high_bits_reference - high_bits
    )
    print(np.sum(diff == 0))
    assert np.sum(diff == 0) >= inner_size - n_allow_err
