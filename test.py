import time

import deai
import numpy as np


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


if __name__ == "__main__":
    # Setup
    vec_length = 10
    values = np.ones((vec_length,), dtype=np.uint32)
    other_values = np.arange(vec_length, dtype=np.uint32)

    # Running everything with timings
    with Timing("keygen"):
        pkey, ckey = deai.create_private_key()
    with Timing("serialization compression key"):
        serialized_compression_key = ckey.serialize()
    with Timing("serialization compression key"):
        compression_key = deai.CompressionKey.deserialize(serialized_compression_key)
    with Timing("encryption"):
        ciphertext = deai.encrypt(pkey, values)
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
        decrypted_result = deai.decrypt(encrypted_result, pkey)

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
