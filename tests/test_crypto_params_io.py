import pytest
import concrete_ml_extensions as fhext
import json
import numpy as np


def test_crypto_params_load():
    with pytest.raises(ValueError):
        json_str = """{
            "xyz_____its_reserved_for_computation": 12,
            "glwe_encryption_noise_distribution_stdev": 0.000000002,
            "encryption_glwe_dimension": 1,
            "polynomial_size": 2048,
            "ciphertext_modulus_bit_count": 63,
            "input_storage_ciphertext_modulus": 39,
            "packing_ks_level": 2, 
            "packing_ks_base_log": 14,
            "packing_ks_polynomial_size": 2048,              
            "packing_ks_glwe_dimension": 1,       
            "output_storage_ciphertext_modulus": 26
        }"""

        fhext.MatmulCryptoParameters.deserialize(json_str)

    json_str = fhext.default_params()
    fhext.MatmulCryptoParameters.deserialize(json_str)


def test_crypto_params_save():
    crypto_params = fhext.MatmulCryptoParameters.deserialize(fhext.default_params())
    params_json = json.loads(fhext.default_params())
    str_out = crypto_params.serialize()
    params_json_rs = json.loads(str_out)

    assert params_json.keys() == params_json_rs.keys(), "Dictionary keys do not match"

    for key in params_json:
        assert np.isclose(
            params_json[key], params_json_rs[key], rtol=1e-9, atol=0
        ), f"Values for key '{key}' are not close enough: {params_json[key]} != {params_json_rs[key]}"
