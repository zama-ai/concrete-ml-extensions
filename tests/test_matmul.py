import time

import concrete_ml_extensions as deai
import numpy as np

PARAMS_8B_2048 = """{
        "bits_reserved_for_computation": 26,
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
    }"""

class Timing:
    def __init__(self, message=""):
        self.message = message

    def __enter__(
        self,
    ):
        print(f"Starting {self.message}")
        self.start = time.time()

    def __exit__(
        self,
        *args,
        **kwargs,
    ):
        end = time.time()
        print(f"{self.message} done in {end - self.start} seconds")


def test_full_dot_product():
    # Setup
    vec_length = 2048
    values = np.ones((vec_length,), dtype=np.uint64)
    other_values = np.arange(vec_length, dtype=np.uint64)

    crypto_params = deai.MatmulCryptoParameters.deserialize(PARAMS_8B_2048)

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
        decrypted_result = deai.decrypt(encrypted_result, pkey, crypto_params, packed_glwe_values=1)

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


def test_matrix_multiplication():
    # Setup
    size = 2049
    packed_glwe_values = size % 2048

    matrix_shape = (1, size)
    values = np.random.randint(0, 2**8, size=matrix_shape, dtype=np.uint64)
    other_matrix = np.random.randint(0, 2**8, size=(size, size), dtype=np.uint64)

    crypto_params = deai.MatmulCryptoParameters.deserialize(PARAMS_8B_2048)

    # Running everything with timings
    with Timing("keygen"):
        pkey, ckey = deai.create_private_key(crypto_params)
    with Timing("serialization compression key"):
        serialized_compression_key = ckey.serialize()
    with Timing("deserialization compression key"):
        compression_key = deai.CompressionKey.deserialize(serialized_compression_key)
    with Timing("encryption"):
        encrypted_matrix = deai.encrypt_matrix(pkey=pkey, crypto_params=crypto_params, data=values)

    with Timing("matrix multiplication"):
        matmul_result = deai.matrix_multiplication(
            encrypted_matrix=encrypted_matrix, data=other_matrix.T, compression_key=compression_key
        )

    with Timing("decryption"):
        decrypted_result = deai.decrypt_matrix(
            matmul_result, pkey, crypto_params, packed_glwe_values=packed_glwe_values
        )

    print("Matrix multiplication encryption test passed")

    # Expected result using numpy
    expected_result = np.dot(values, other_matrix.T)

    print(decrypted_result.shape)
    print(expected_result.shape)
    assert (
        decrypted_result.shape == expected_result.shape
    ), "Decrypted matrix shape mismatch"

    # Extract the 12 MSB from both results
    msb_decrypted = decrypted_result >> (26 - 12)
    msb_expected = expected_result >> (26 - 12)

    print(msb_decrypted)
    print(msb_expected)
    np.testing.assert_array_equal(
        msb_decrypted,
        msb_expected,
        "The 12 MSB of the decrypted matrix do not match the expected result",
    )
    print("Encrypted matrix multiplication matches the original numpy dot product for the 12 MSB")


if __name__ == "__main__":
    test_full_dot_product()
    test_matrix_multiplication()
