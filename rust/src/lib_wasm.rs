#![allow(clippy::excessive_precision)]
use tfhe::ClientKey;
use tfhe::safe_serialization::{safe_deserialize, safe_serialize};
use tfhe::Seed;
use crate::radix_utils::{core_encrypt_u64_radix_array, core_decrypt_u64_radix_array, core_keygen_radix};

use wasm_bindgen::prelude::*;
use js_sys::{BigUint64Array, Uint8Array};
use std::io::Cursor;

use tfhe::core_crypto::seeders::Seeder;
use getrandom::getrandom;

#[derive(Clone, Copy, Debug)]
#[wasm_bindgen]
pub struct JsSeeder;

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
    let client_key: ClientKey = safe_deserialize(&mut Cursor::new(&client_key_ser), SERIALIZE_SIZE_LIMIT)
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
    let client_key: ClientKey = safe_deserialize(&mut Cursor::new(&client_key_ser), SERIALIZE_SIZE_LIMIT)
        .map_err(|e| JsValue::from_str(&format!("ClientKey deserialization error: {}", e)))?;

    let encrypted_data_ser = encrypted_data_js.to_vec();
    
    core_decrypt_u64_radix_array(&encrypted_data_ser, &client_key)
        .map_err(|e| JsValue::from_str(&format!("Core decryption error: {}", e)))
        .map(|results_flat| BigUint64Array::from(results_flat.as_slice()))
}