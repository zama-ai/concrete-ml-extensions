use tfhe::prelude::*;
use tfhe::{ClientKey, FheUint64, ServerKey, ConfigBuilder, Seed};
use tfhe::core_crypto::prelude::Seeder;

#[cfg(target_arch = "wasm32")]
fn get_wasm_seed() -> Seed {
    let mut seeder = crate::JsSeeder;
    seeder.seed()
}

fn generate_keys_from_seed(seed: Seed) -> (ClientKey, ServerKey) {
    let config = ConfigBuilder::default()
        .use_custom_parameters(tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS)
        .build();

    let client_key = ClientKey::generate_with_seed(config, seed);
    let server_key = client_key.generate_server_key();
    (client_key, server_key)
}

pub(crate) fn core_keygen_radix() -> (ClientKey, ServerKey) {
    #[cfg(target_arch = "wasm32")]
    let seed = get_wasm_seed();

    #[cfg(not(target_arch = "wasm32"))]
    let seed = {
        let mut seeder = tfhe::core_crypto::seeders::new_seeder();
        seeder.seed()
    };

    generate_keys_from_seed(seed)
}

pub(crate) fn core_encrypt_u64_radix_array(
    values: &[u64],
    client_key: &ClientKey,
) -> Result<Vec<u8>, bincode::Error> {
    let cts: Vec<FheUint64> = values
        .iter()
        .map(|v| FheUint64::encrypt(*v, client_key))
        .collect();
    bincode::serialize(&cts)
}

pub(crate) fn core_decrypt_u64_radix_array(
    serialized_cts: &[u8],
    client_key: &ClientKey,
) -> Result<Vec<u64>, bincode::Error> {
    let fhe_array: Vec<FheUint64> = bincode::deserialize(serialized_cts)?;
    Ok(fhe_array.iter().map(|v| v.decrypt(client_key)).collect())
}