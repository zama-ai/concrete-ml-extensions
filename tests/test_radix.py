import numpy as np
import concrete_ml_extensions as fhext
import pytest


@pytest.mark.parametrize("ndims", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.uint16, np.int16, np.uint64])
def test_radix_encrypt_decrypt_keygen(ndims, dtype):

    vmin = np.iinfo(dtype).min
    vmax = np.iinfo(dtype).max
    shape = tuple(map(int, np.random.randint(1, 5, size=(ndims,))))

    arr = np.random.randint(vmin, vmax, size=shape, dtype=dtype)
    sk, _, _ = fhext.keygen_radix()

    # pylint: disable=no-member
    dtype_bytes = np.dtype(dtype).itemsize
    # Check if encryption is supported for this dtype and shape
    if len(shape) == 2 and dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint64):
        blob = fhext.encrypt_radix(arr, sk)
        # Determine bitwidth based on dtype
        bitwidth = 8 * dtype_bytes
        is_signed = np.issubdtype(np.dtype(dtype), np.signedinteger)

        # Check if decryption is supported for this bitwidth and sign
        if bitwidth in (8, 16) or (bitwidth == 64 and not is_signed):
            arr_out = fhext.decrypt_radix(
                blob,
                arr.shape,
                bitwidth,
                is_signed,
                sk,
            )
            assert np.all(arr_out == arr)
        else:
            # Expect decryption to fail for unsupported types (e.g., int64)
            with pytest.raises(
                AssertionError,
                match=".*Cannot decrypt datatype.*|.*not currently supported.*",
            ):
                fhext.decrypt_radix(blob, arr.shape, bitwidth, is_signed, sk)

    else:
        with pytest.raises(AssertionError, match=".*Cannot encrypt datatype.*"):
            fhext.encrypt_radix(arr, sk)
