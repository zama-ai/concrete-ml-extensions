#![allow(clippy::excessive_precision)]

use serde::{Deserialize, Serialize};
use tfhe::core_crypto::prelude;
use tfhe::core_crypto::prelude::*;
use std::marker::PhantomData;

use crate::compression;
use crate::encryption;
use crate::ml;

type Scalar = u64;
uniffi::setup_scaffolding!();


// ===== Test functions =====

#[uniffi::export]
pub fn say_hello() {
    println!("Hello from lib2!");
}

#[uniffi::export]
pub fn add(a: u64, b: u64) -> u64 {
    a + b
}


// ===== Private Key =====

#[derive(Serialize, Deserialize)]
#[derive(uniffi::Object)]
struct PrivateKey {
    inner: prelude::GlweSecretKey<Vec<Scalar>>,
    post_compression_secret_key: GlweSecretKey<Vec<Scalar>>,
}

#[uniffi::export]
impl PrivateKey {
    fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).unwrap()
    }
}

/// Deserialize from a byte vector into a `PrivateKey` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn PrivateKey_deserialize(content: Vec<u8>) -> PrivateKey {
    bincode::deserialize(&content).unwrap()
}



// ===== MatmulCryptoParameters =====

#[derive(Serialize, Deserialize)]
#[derive(uniffi::Object)]
struct MatmulCryptoParameters {
    // Global parameters
    ciphertext_modulus_bit_count: usize,  // 64?
    bits_reserved_for_computation: usize, // for encoding, related to poly size ?

    // Input parameters
    encryption_glwe_dimension: GlweDimension,      // k_in
    polynomial_size: usize,                        // N_in
    input_storage_ciphertext_modulus: usize,       // q_in
    glwe_encryption_noise_distribution_stdev: f64, // computed with RO

    // Output parameters
    packing_ks_level: DecompositionLevelCount, // l_pks
    packing_ks_base_log: DecompositionBaseLog, // log_b_pks
    packing_ks_polynomial_size: usize,         // N_out
    packing_ks_glwe_dimension: GlweDimension,  // k_out
    output_storage_ciphertext_modulus: usize,  // q_out
    pks_noise_distrubution_stdev: f64,         // computed  with RO
}

#[uniffi::export]
impl MatmulCryptoParameters {
    /// Serialize into a JSON string.
    fn serialize(&self) -> Result<String, String> {
        serde_json::to_string(&self).map_err(|e| format!("Can not serialize crypto-parameters {}", e))
    }
}

/// Deserialize from a JSON string into a `MatmulCryptoParameters` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn MatmulCryptoParameters_deserialize(content: String) -> Result<MatmulCryptoParameters, String> {
    serde_json::from_str(&content).map_err(|e| format!("Can not deserialize cryptoparameters {}", e))
}



// ===== CpuCompressionKey =====

#[derive(Serialize, Deserialize)]
#[derive(uniffi::Object)]
struct CpuCompressionKey {
    inner: compression::CompressionKey<Scalar>,
    buffers: compression::CpuCompressionBuffers<Scalar>,
}

#[uniffi::export]
impl CpuCompressionKey {
    /// Serialize into a byte vector.
    fn serialize(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(&self.inner).map_err(|e| format!("Failed to serialize: {}", e))
    }
}

/// Deserialize from a byte vector into a `CpuCompressionKey` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn CpuCompressionKey_deserialize(content: Vec<u8>) -> Result<CpuCompressionKey, String> {
    bincode::deserialize(&content).map_err(|e| format!("Failed to deserialize: {}", e))
}



// ===== CipherText =====

#[derive(Serialize, Deserialize, Clone)]
#[derive(uniffi::Object)]
struct CipherText {
    inner: ml::SeededCompressedEncryptedVector<Scalar>,
}

#[uniffi::export]
impl CipherText {
    /// Serialize into a byte vector.
    fn serialize(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(&self).map_err(|e| format!("Failed to serialize: {}", e))
    }
}

/// Deserialize from a byte vector into a `CipherText` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn CipherText_deserialize(content: Vec<u8>) -> Result<CipherText, String> {
    bincode::deserialize(&content).map_err(|e| format!("Failed to deserialize: {}", e))
}



// ===== EncryptedMatrix =====
#[derive(Serialize, Deserialize, Clone)]
#[derive(uniffi::Object)]
pub struct EncryptedMatrix {
    pub inner: Vec<ml::SeededCompressedEncryptedVector<Scalar>>,
    pub shape: (usize, usize),
}

#[uniffi::export]
impl EncryptedMatrix {
    fn serialize(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(&self).map_err(|e| format!("Failed to serialize: {}", e))
    }
}

/// Deserialize from a byte vector into an `EncryptedMatrix` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn EncryptedMatrix_deserialize(content: Vec<u8>) -> Result<CipherText, String> {
    bincode::deserialize(&content).map_err(|e| format!("Failed to deserialize: {}", e))
}



// ===== CompressedResultCipherText =====

#[derive(Serialize, Deserialize, Clone)]
#[derive(uniffi::Object)]
struct CompressedResultCipherText {
    inner: Vec<prelude::compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext<Scalar>>,
}



// ===== CompressedResultEncryptedMatrix =====

#[derive(Serialize, Deserialize, Clone)]
#[derive(uniffi::Object)]
struct CompressedResultEncryptedMatrix {
    inner: Vec<CompressedResultCipherText>,
}

#[uniffi::export]
impl CompressedResultEncryptedMatrix {
    fn serialize(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(&self).map_err(|e| format!("Failed to serialize: {}", e))
    }
}

/// Deserialize from a byte vector into an `CompressedResultEncryptedMatrix` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn CompressedResultEncryptedMatrix_deserialize(content: Vec<u8>) -> Result<CompressedResultEncryptedMatrix, String> {
    bincode::deserialize(&content).map_err(|e| format!("Failed to deserialize: {}", e))
}



// ===== Key Gen =====

fn create_private_key_internal(
    crypto_params: &MatmulCryptoParameters,
) -> (
    GlweSecretKey<Vec<Scalar>>,
    GlweSecretKey<Vec<Scalar>>,
    compression::CompressionKey<Scalar>,
) {
    // This could be a method to generate a private key object
    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    let mut secret_rng = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());

    let compression_params = compression::CompressionKeyParameters::<Scalar> {
        packing_ks_level: crypto_params.packing_ks_level,
        packing_ks_base_log: crypto_params.packing_ks_base_log,
        packing_ks_polynomial_size: PolynomialSize(crypto_params.packing_ks_polynomial_size),
        packing_ks_glwe_dimension: crypto_params.packing_ks_glwe_dimension,
        lwe_per_glwe: LweCiphertextCount(crypto_params.polynomial_size),
        packing_ciphertext_modulus: CiphertextModulus::try_new_power_of_2(
            crypto_params.ciphertext_modulus_bit_count,
        )
        .unwrap(),
        storage_log_modulus: CiphertextModulusLog(crypto_params.output_storage_ciphertext_modulus),
        packing_ks_key_noise_distribution: DynamicDistribution::new_gaussian_from_std_dev(
            StandardDev(crypto_params.pks_noise_distrubution_stdev),
        ),
    };
    let glwe_secret_key: GlweSecretKey<Vec<Scalar>> =
        allocate_and_generate_new_binary_glwe_secret_key(
            crypto_params.encryption_glwe_dimension,
            PolynomialSize(crypto_params.polynomial_size),
            &mut secret_rng,
        );
    let glwe_secret_key_as_lwe_secret_key = glwe_secret_key.as_lwe_secret_key();

    let (post_compression_glwe_secret_key, compression_key) =
        compression::CompressionKey::new(&glwe_secret_key_as_lwe_secret_key, compression_params);

    (
        glwe_secret_key,
        post_compression_glwe_secret_key,
        compression_key,
    )
}

// #[uniffi::export]
fn cpu_create_private_key(
    crypto_params: &MatmulCryptoParameters,
) -> (PrivateKey, CpuCompressionKey) {
    let (glwe_secret_key, post_compression_glwe_secret_key, compression_key) =
        create_private_key_internal(crypto_params);

    return (
        PrivateKey {
            inner: glwe_secret_key,
            post_compression_secret_key: post_compression_glwe_secret_key,
        },
        CpuCompressionKey {
            inner: compression_key,
            buffers: compression::CpuCompressionBuffers::<Scalar> { _tmp: PhantomData },
        },
    );
}

fn internal_encrypt(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: &[Scalar],
) -> Result<CipherText, String> {
    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    
    let ciphertext_modulus = CiphertextModulus::try_new_power_of_2(crypto_params.ciphertext_modulus_bit_count).unwrap();
    
    let glwe_encryption_noise_distribution = prelude::DynamicDistribution::new_gaussian_from_std_dev(StandardDev(
        crypto_params.glwe_encryption_noise_distribution_stdev,
    ));
    
    let seeded_encrypted_vector = ml::SeededCompressedEncryptedVector::<Scalar>::new(
        &data,
        &pkey.inner,
        crypto_params.bits_reserved_for_computation,
        CiphertextModulusLog(crypto_params.input_storage_ciphertext_modulus),
        glwe_encryption_noise_distribution,
        ciphertext_modulus,
        seeder,
    );
    Ok(CipherText {
        inner: seeded_encrypted_vector,
    })
}

fn internal_decrypt(
    compressed_result: &CompressedResultCipherText,
    crypto_params: &MatmulCryptoParameters,
    private_key: &PrivateKey,
    num_valid_glwe_values_in_last_ciphertext: usize,
) -> Result<Vec<Scalar>, String> {
    let mut decrypted_result = Vec::new();
    let last_index = compressed_result.inner.len() - 1;

    for (index, compressed) in compressed_result.inner.iter().enumerate() {
        let extracted = compressed.extract();
        let decrypted_dot: Vec<Scalar> = encryption::decrypt_glwe(
            &private_key.post_compression_secret_key,
            &extracted,
            crypto_params.bits_reserved_for_computation,
        );

        if index == last_index {
            decrypted_result.extend(
                decrypted_dot
                    .into_iter()
                    .take(num_valid_glwe_values_in_last_ciphertext),
            );
        } else {
            decrypted_result.extend(decrypted_dot);
        }
    }

    Ok(decrypted_result)
}

#[uniffi::export]
fn encrypt_matrix(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: Vec<Vec<Scalar>>,
) -> Result<EncryptedMatrix, String> {
    let mut encrypted_matrix = Vec::new();
    for row in data.iter() {
        let row_array = row.clone();
        let encrypted_row = internal_encrypt(pkey, crypto_params, row_array.as_slice())?;
        encrypted_matrix.push(encrypted_row.inner);
    }
    
    Ok(EncryptedMatrix {
        inner: encrypted_matrix,
        shape: (data.len(), data[0].len()),
    })
}

// #[uniffi::export]
fn decrypt_matrix(
    compressed_matrix: CompressedResultEncryptedMatrix,
    private_key: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    num_valid_glwe_values_in_last_ciphertext: u64, // Change `usize` to `u64`
) -> Result<Vec<Vec<Scalar>>, String> {
    let decrypted_matrix: Vec<Vec<Scalar>> = compressed_matrix
        .inner
        .iter()
        .map(|compressed_row| {
            internal_decrypt(
                compressed_row,
                crypto_params,
                private_key,
                num_valid_glwe_values_in_last_ciphertext as usize, // Converting back just in case
            )
        })
        .collect::<Result<_, _>>()
        .map_err(|e| format!("Decryption failed: {}", e))?;

    Ok(decrypted_matrix)
}

static PARAMS_8B_2048_NEW: &str = r#"{
    "bits_reserved_for_computation": 27,
    "glwe_encryption_noise_distribution_stdev": 8.67361737996499e-19,
    "encryption_glwe_dimension": 1,
    "polynomial_size": 2048,
    "ciphertext_modulus_bit_count": 32,
    "input_storage_ciphertext_modulus": 32,
    "packing_ks_level": 1,
    "packing_ks_base_log": 21,
    "packing_ks_polynomial_size": 2048,
    "packing_ks_glwe_dimension": 1,
    "output_storage_ciphertext_modulus": 19,
    "pks_noise_distrubution_stdev": 8.095547030480235e-30
}"#;

#[uniffi::export]
fn default_params() -> String {
    PARAMS_8B_2048_NEW.to_string()
}


// === Lift Trait Implementation (for UniFFI) ===
// • Avoid usize in UniFFI-exposed functions.
// • Convert usize to u64/u32 when passing values across FFI boundaries.
// • Use .as usize inside Rust code if needed.
