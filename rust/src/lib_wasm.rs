#![allow(clippy::excessive_precision)]
use crate::radix_utils::{
    core_decrypt_u64_radix_array, core_encrypt_u64_radix_array, core_keygen_radix,
};
use tfhe::safe_serialization::{safe_deserialize, safe_serialize};
use tfhe::{ClientKey, Seed};

use js_sys::{BigUint64Array, Uint8Array};
use std::io::Cursor;
use wasm_bindgen::prelude::*;

use getrandom::getrandom;
use tfhe::core_crypto::seeders::Seeder;

use crate::fhext_classes::*;

#[derive(Clone, Copy, Debug)]
#[wasm_bindgen]
pub struct JsSeeder;

#[wasm_bindgen]
pub struct WasmPrivateKey(pub(crate) PrivateKey);

#[wasm_bindgen]
pub struct WasmMatmulCryptoParameters(pub(crate) MatmulCryptoParameters);

#[wasm_bindgen]
pub struct WasmEncryptedMatrix(pub(crate) EncryptedMatrix);

#[wasm_bindgen]
pub struct WasmCompressionKey(pub(crate) EncryptedMatrix);

#[wasm_bindgen]
pub struct WasmCompressedResultEncryptedMatrix(pub(crate) CompressedResultEncryptedMatrix);

impl MatmulCryptoParameters {
    fn serialize(&self) -> Result<String, JsValue> {
        return match serde_json::to_string(&self) {
            Ok(json_str) => Ok(json_str),
            Err(error) => Err(JsValue::from_str(&format!(
                "Can not serialize crypto-parameters {error}"
            ))),
        };
    }

    fn deserialize(content: &String) -> Result<MatmulCryptoParameters, JsValue> {
        return match serde_json::from_str(&content.to_string()) {
            Ok(p) => Ok(p),
            Err(error) => Err(JsValue::from_str(&format!(
                "Can not deserialize cryptoparameters {error}"
            ))),
        };
    }
}

impl Seeder for JsSeeder {
    fn seed(&mut self) -> Seed {
        let mut bytes = [0u8; 16];
        getrandom(&mut bytes).expect("js crypto rng failure");
        Seed(u128::from_le_bytes(bytes))
    }

    fn is_available() -> bool {
        true
    }
}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    Ok(())
}

const SERIALIZE_SIZE_LIMIT: u64 = 1_000_000_000;

#[wasm_bindgen]
pub fn keygen_radix_u64_wasm() -> Result<JsValue, JsValue> {
    let (client_key, server_key) = core_keygen_radix();

    let mut client_bytes = Vec::new();
    safe_serialize(&client_key, &mut client_bytes, SERIALIZE_SIZE_LIMIT)
        .map_err(|e| JsValue::from_str(&format!("ClientKey serialization error: {}", e)))?;

    let mut server_bytes = Vec::new();
    safe_serialize(&server_key, &mut server_bytes, SERIALIZE_SIZE_LIMIT)
        .map_err(|e| JsValue::from_str(&format!("ServerKey serialization error: {}", e)))?;

    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &"clientKey".into(),
        &Uint8Array::from(client_bytes.as_slice()),
    )?;
    js_sys::Reflect::set(
        &result,
        &"serverKey".into(),
        &Uint8Array::from(server_bytes.as_slice()),
    )?;

    Ok(result.into())
}

#[wasm_bindgen]
pub fn encrypt_serialize_u64_radix_flat_wasm(
    value_flat_js: &BigUint64Array,
    client_key_ser_js: &Uint8Array,
) -> Result<Uint8Array, JsValue> {
    let client_key_ser = client_key_ser_js.to_vec();
    let client_key: ClientKey =
        safe_deserialize(&mut Cursor::new(&client_key_ser), SERIALIZE_SIZE_LIMIT)
            .map_err(|e| JsValue::from_str(&format!("ClientKey deserialization error: {}", e)))?;

    let mut data_vec_flat: Vec<u64> = vec![0; value_flat_js.length() as usize];
    value_flat_js.copy_to(&mut data_vec_flat);

    core_encrypt_u64_radix_array(&data_vec_flat, &client_key)
        .map_err(|e| JsValue::from_str(&format!("Core encryption error: {}", e)))
        .map(|serialized_cts| Uint8Array::from(serialized_cts.as_slice()))
}

#[wasm_bindgen]
pub fn decrypt_serialized_u64_radix_flat_wasm(
    encrypted_data_js: &Uint8Array,
    client_key_ser_js: &Uint8Array,
) -> Result<BigUint64Array, JsValue> {
    let client_key_ser = client_key_ser_js.to_vec();
    let client_key: ClientKey =
        safe_deserialize(&mut Cursor::new(&client_key_ser), SERIALIZE_SIZE_LIMIT)
            .map_err(|e| JsValue::from_str(&format!("ClientKey deserialization error: {}", e)))?;

    let encrypted_data_ser = encrypted_data_js.to_vec();

    core_decrypt_u64_radix_array(&encrypted_data_ser, &client_key)
        .map_err(|e| JsValue::from_str(&format!("Core decryption error: {}", e)))
        .map(|results_flat| BigUint64Array::from(results_flat.as_slice()))
}

#[wasm_bindgen]
pub fn default_params(
    bits_reserved_for_computation: usize,
) -> Result<WasmMatmulCryptoParameters, JsValue> {
    let mut result: MatmulCryptoParameters = serde_json::from_str(PARAMS_8B_2048_NEW).unwrap();
    result.bits_reserved_for_computation = bits_reserved_for_computation;
    Ok(WasmMatmulCryptoParameters { 0: result })
}

#[wasm_bindgen]
pub fn encrypt_matrix(
    pkey: &WasmPrivateKey,
    crypto_params: &WasmMatmulCryptoParameters,
    data: &JsValue,
    dims: &JsValue,
) -> Result<WasmEncryptedMatrix, JsValue> {
    let mut encrypted_matrix = Vec::new();

    let data_vec: Vec<Vec<u64>> = data.into_serde().unwrap();

    for row in data_vec.iter() {
        let row_array = row.to_owned();
        let encrypted_row = internal_encrypt(&pkey.0, &crypto_params.0, row_array.as_slice());
        encrypted_matrix.push(encrypted_row.inner);
    }

    let dims_vec: Vec<usize> = dims.into_serde().unwrap();

    Ok(WasmEncryptedMatrix {
        0: EncryptedMatrix {
            inner: encrypted_matrix,
            shape: (dims_vec[0] as usize, dims_vec[1] as usize),
        },
    })
}

#[wasm_bindgen]
pub fn create_matmul_private_key(
    crypto_params: &WasmMatmulCryptoParameters,
) -> Result<JsValue, JsValue> {
    let (glwe_secret_key, post_compression_glwe_secret_key, compression_key) =
        create_private_key_internal(&crypto_params.0);

    let pk = PrivateKey {
        inner: glwe_secret_key,
        post_compression_secret_key: post_compression_glwe_secret_key,
    };

    let client_bytes: Vec<u8> = bincode::serialize(&pk)
        .map_err(|e| JsValue::from_str(&format!("ClientKey serialization error: {}", e)))?;

    let server_key = CpuCompressionKey {
        inner: compression_key,
    };

    let server_bytes: Vec<u8> = bincode::serialize(&server_key)
        .map_err(|e| JsValue::from_str(&format!("ServerKey serialization error: {}", e)))?;

    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &"clientKey".into(),
        &Uint8Array::from(client_bytes.as_slice()),
    )?;
    js_sys::Reflect::set(
        &result,
        &"serverKey".into(),
        &Uint8Array::from(server_bytes.as_slice()),
    )?;

    return Ok(result.into());
}

#[wasm_bindgen]
pub fn decrypt_matrix_u64(
    compressed_matrix: WasmCompressedResultEncryptedMatrix,
    private_key: &WasmPrivateKey,
    crypto_params: &WasmMatmulCryptoParameters,
    num_valid_glwe_values_in_last_ciphertext: usize,
) -> Result<JsValue, JsValue> {
    let decrypted_matrix: Vec<Vec<u64>> = compressed_matrix
        .0
        .inner
        .iter()
        .map(|compressed_row| {
            internal_decrypt(
                compressed_row,
                &crypto_params.0,
                &private_key.0,
                num_valid_glwe_values_in_last_ciphertext,
            )
        })
        .collect::<Vec<_>>();

    let result_js = js_sys::Array::new();
    for v in decrypted_matrix {
        result_js.push(&BigUint64Array::from(v.as_slice()));
    }
    Ok(result_js.into())
}
