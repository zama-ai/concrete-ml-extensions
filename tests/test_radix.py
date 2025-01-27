import numpy as np
import concrete_ml_extensions as fhext
import pytest

@pytest.mark.parametrize("ndims", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.uint16, np.int16])
def test_radix_encrypt_decrypt_keygen(ndims, dtype):

    vmin = np.iinfo(dtype).min
    vmax = np.iinfo(dtype).max
    shape = tuple(map(int, np.random.randint(1, 5, size=(ndims,))))

    arr = np.random.randint(vmin, vmax, size=shape, dtype=dtype)
    sk, _, _ = fhext.keygen_radix()  # pylint: disable=no-member
    dtype_bytes = np.dtype(dtype).itemsize == 1
    if len(shape) == 2 and dtype_bytes == 1:
        blob = fhext.encrypt_radix(arr, sk)
        arr_out = fhext.decrypt_radix(blob, 
            arr.shape, 
            8 * dtype_bytes, 
            np.issubdtype(np.dtype(dtype), np.signedinteger), 
            sk
        )

        assert np.all(arr_out == arr)
    else:
        with pytest.raises(AssertionError, match=".*Cannot encrypt datatype.*"):
            fhext.encrypt_radix(arr, sk)