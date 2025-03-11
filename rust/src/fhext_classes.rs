#[cfg(feature = "python_bindings")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use tfhe::core_crypto::prelude;
use tfhe::core_crypto::prelude::*;

use crate::{compression, encryption, ml, Scalar};

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "swift_bindings", derive(uniffi::Object))]
#[cfg_attr(feature = "python_bindings", pyclass)]
pub struct PrivateKey {
    pub inner: prelude::GlweSecretKey<Vec<Scalar>>,
    pub post_compression_secret_key: GlweSecretKey<Vec<Scalar>>,
}

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python_bindings", pyclass)]
#[cfg_attr(feature = "swift_bindings", derive(uniffi::Object))]
pub struct EncryptedMatrix {
    pub inner: Vec<ml::SeededCompressedEncryptedVector<Scalar>>,
    pub shape: (usize, usize),
}

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python_bindings", pyclass)]
#[cfg_attr(feature = "swift_bindings", derive(uniffi::Object))]
pub struct MatmulCryptoParameters {
    // Global parameters
    pub(crate) ciphertext_modulus_bit_count: usize, // 64?
    pub(crate) bits_reserved_for_computation: usize, // for encoding, related to poly size ?

    // Input parameters
    pub(crate) encryption_glwe_dimension: GlweDimension, // k_in
    pub(crate) polynomial_size: usize,                   // N_in
    pub(crate) input_storage_ciphertext_modulus: usize,  // q_in
    pub(crate) glwe_encryption_noise_distribution_stdev: f64, // computed with RO

    // Output parameters
    pub(crate) packing_ks_level: DecompositionLevelCount, // l_pks
    pub(crate) packing_ks_base_log: DecompositionBaseLog, // log_b_pks
    pub(crate) packing_ks_polynomial_size: usize,         // N_out
    pub(crate) packing_ks_glwe_dimension: GlweDimension,  // k_out
    pub(crate) output_storage_ciphertext_modulus: usize,  // q_out
    pub(crate) pks_noise_distrubution_stdev: f64,         // computed  with RO
}

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python_bindings", pyclass)]
#[cfg_attr(feature = "swift_bindings", derive(uniffi::Object))]

pub struct CpuCompressionKey {
    pub(crate) inner: compression::CompressionKey<Scalar>,
    //    pub(crate) buffers: compression::CpuCompressionBuffers<Scalar>,
}

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python_bindings", pyclass)]
#[cfg_attr(feature = "swift_bindings", derive(uniffi::Object))]

pub struct CipherText {
    pub(crate) inner: crate::ml::SeededCompressedEncryptedVector<Scalar>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CompressedResultCipherText {
    pub(crate) inner: Vec<prelude::compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext<Scalar>>,
}

/*#[derive(Serialize, Deserialize, Clone)]
#[derive(uniffi::Object)]
struct CompressedResultCipherText {
    inner: Vec<prelude::compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext<Scalar>>,
}*/

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python_bindings", pyclass)]
#[cfg_attr(feature = "swift_bindings", derive(uniffi::Object))]
pub struct CompressedResultEncryptedMatrix {
    pub(crate) inner: Vec<CompressedResultCipherText>,
}

pub static PARAMS_8B_2048_NEW: &str = r#"{
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

pub fn create_private_key_internal(
    crypto_params: &MatmulCryptoParameters,
) -> (
    GlweSecretKey<Vec<Scalar>>,
    GlweSecretKey<Vec<Scalar>>,
    compression::CompressionKey<Scalar>,
) {
    // This could be a method to generate a private key object
    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    let mut secret_rng = SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());
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

pub fn internal_encrypt(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: &[Scalar],
) -> CipherText {
    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    let ciphertext_modulus: CiphertextModulus<Scalar> =
        CiphertextModulus::try_new_power_of_2(crypto_params.ciphertext_modulus_bit_count).unwrap();
    let glwe_encryption_noise_distribution: prelude::DynamicDistribution<Scalar> =
        DynamicDistribution::new_gaussian_from_std_dev(StandardDev(
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
    CipherText {
        inner: seeded_encrypted_vector,
    }
}

pub fn internal_decrypt(
    compressed_result: &CompressedResultCipherText,
    crypto_params: &MatmulCryptoParameters,
    private_key: &PrivateKey,
    num_valid_glwe_values_in_last_ciphertext: usize,
) -> Vec<Scalar> {
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

    decrypted_result
}
