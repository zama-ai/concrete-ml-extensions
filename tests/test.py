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
        decrypted_result = deai.decrypt(encrypted_result, pkey, crypto_params)

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

if __name__ == "__main__":
    test_full_dot_product()