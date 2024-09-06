import pytest
import concrete_ml_extensions as deai
import json
from conftest import PARAMS_8B_2048

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

        deai.MatmulCryptoParameters.deserialize(json_str)

    json_str = json.dumps(PARAMS_8B_2048)
    deai.MatmulCryptoParameters.deserialize(json_str)


def test_crypto_params_save(crypto_params):
    params_json = PARAMS_8B_2048
    str_out = crypto_params.serialize()
    params_json_rs = json.loads(str_out)

    assert params_json == params_json_rs
