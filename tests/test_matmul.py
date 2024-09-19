import time
import concrete_ml_extensions as deai
import numpy as np
import pytest
from conftest import Timing, PARAMS_8B_2048
import json


@pytest.mark.parametrize("size", [128, 512, 2048, 4096, 8192])
def test_full_dot_product(size, crypto_params):
    # Setup
    vec_length = size
    num_valid_glwe_values_in_last_ciphertext = size % 2048
    values = np.ones((vec_length,), dtype=np.uint64)
    other_values = np.arange(vec_length, dtype=np.uint64)

    # Running everything with timings
    with Timing("keygen"):
        pkey, ckey = deai.create_private_key(crypto_params)
    with Timing("serialization compression key"):
        serialized_compression_key = ckey.serialize()
    with Timing("serialization compression key"):
        compression_key = deai.CompressionKey.deserialize(serialized_compression_key)
    with Timing("encryption"):
        ciphertext = deai.encrypt(pkey, crypto_params, values)
    with Timing("input serialization"):
        serialized_ciphertext = ciphertext.serialize()
    with Timing("input deserialization"):
        ciphertext = deai.CipherText.deserialize(serialized_ciphertext)
    with Timing("dot_prod"):
        encrypted_result = deai.dot_product(ciphertext, other_values, ckey)
    with Timing("output serialization"):
        serialized_encrypted_result = encrypted_result.serialize()
    with Timing("output deserialization"):
        encrypted_result = deai.CompressedResultCipherText.deserialize(
            serialized_encrypted_result
        )
    with Timing("decryption"):
        decrypted_result = deai.decrypt(
            encrypted_result,
            pkey,
            crypto_params,
            num_valid_glwe_values_in_last_ciphertext=1,
        )

    print(
        f"""
         {ciphertext=},
         {pkey=},
         {values=},
         {encrypted_result=},
         {decrypted_result=},
         {np.dot(values, other_values)=},
        """
    )


@pytest.mark.parametrize("size", [512, 1024, 2048, 4096])
def test_matrix_multiplication(size, crypto_params):

    matrix_shape = (1, size)
    values = np.random.randint(0, 2**2, size=matrix_shape, dtype=np.uint64)
    other_matrix = np.random.randint(0, 2**2, size=(size, size), dtype=np.uint64)

    # Expected result using numpy
    expected_result = np.dot(values, other_matrix.T)

    # Calculate the bit-width based on the max value in the expected result
    # In practice we will use the calibration in PTQ to find the right bit-width
    max_value = np.max(expected_result)
    max_bit_width_compute = int(np.ceil(np.log2(max_value + 1)))

    params = json.loads(crypto_params.serialize())
    params["bits_reserved_for_computation"] = (
        max_bit_width_compute + 1
    )  # +1 for sign bit?
    modified_crypto_params = deai.MatmulCryptoParameters.deserialize(json.dumps(params))

    # The number of valid GLWE values in the last ciphertext is the size of the matrix
    # or 2048 if the size is a multiple of 2048
    num_valid_glwe_values_in_last_ciphertext = (
        size % params["polynomial_size"] or params["polynomial_size"]
    )

    # Running everything with timings
    with Timing("keygen"):
        pkey, ckey = deai.create_private_key(modified_crypto_params)
    with Timing("serialization compression key"):
        serialized_compression_key = ckey.serialize()
    with Timing("deserialization compression key"):
        compression_key = deai.CompressionKey.deserialize(serialized_compression_key)
    with Timing("encryption"):
        encrypted_matrix = deai.encrypt_matrix(
            pkey=pkey, crypto_params=modified_crypto_params, data=values
        )

    with Timing("matrix multiplication"):
        matmul_result = deai.matrix_multiplication(
            encrypted_matrix=encrypted_matrix,
            data=other_matrix.T,
            compression_key=compression_key,
        )

    with Timing("decryption"):
        decrypted_result = deai.decrypt_matrix(
            matmul_result,
            pkey,
            modified_crypto_params,
            num_valid_glwe_values_in_last_ciphertext=num_valid_glwe_values_in_last_ciphertext,
        ).astype(np.int64)

    print("Matrix multiplication encryption test passed")

    print(decrypted_result.shape)
    print(expected_result.shape)
    assert (
        decrypted_result.shape == expected_result.shape
    ), "Decrypted matrix shape mismatch"

    # Print bit-width
    print(f"Bit-width expected: {max_bit_width_compute}")

    # print dtype of decrypted_result
    print(decrypted_result.dtype)
    print(expected_result.dtype)

    shift_delta_bits = 12 if max_bit_width_compute <= 12 else max_bit_width_compute - 12
    # Extract the 12 MSB from both results
    msb_decrypted = decrypted_result >> shift_delta_bits
    msb_expected = expected_result >> shift_delta_bits

    # Compare the arrays and find diverging values
    diverging_indices = np.where(msb_decrypted != msb_expected)
    total_elements = msb_decrypted.size
    mismatch_count = len(diverging_indices[0])
    mismatch_percentage = (mismatch_count / total_elements) * 100

    print("Percentage of mismatches:", mismatch_percentage)
    if mismatch_percentage > 1:
        print("\nDiverging values found:")
        for idx in zip(*diverging_indices):
            print(f"Index {idx}:")
            print(
                f"  Original: Decrypted = {decrypted_result[idx]}, Expected = {expected_result[idx]}"
            )
            print(
                f"  MSB:      Decrypted = {msb_decrypted[idx]}, Expected = {msb_expected[idx]}"
            )
            print(
                f"  Absolute Difference: Original = {abs(int(decrypted_result[idx]) - int(expected_result[idx]))}, "
                f"MSB = {abs(int(msb_decrypted[idx]) - int(msb_expected[idx]))}"
            )
            print()

        error_message = (
            f"The 12 MSB of the decrypted matrix do not match the expected result.\n"
            f"Number of mismatches: {mismatch_count} out of {total_elements} elements.\n"
            f"Percentage of mismatches: {mismatch_percentage:.2f}%"
        )
        print(error_message)
        raise AssertionError(error_message)
    else:
        print(
            "Encrypted matrix multiplication matches the original numpy dot product for the 12 MSB"
        )


print(
    "Encrypted matrix multiplication matches the original numpy dot product for the 12 MSB"
)

if __name__ == "__main__":
    test_full_dot_product()
    test_matrix_multiplication()
