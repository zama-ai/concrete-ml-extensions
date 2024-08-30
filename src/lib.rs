#![allow(clippy::excessive_precision)]

// TODO: Implement something like Ix1 dimension handling for GLWECipherTexts

use numpy::{Ix1, PyReadonlyArray};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyNone, PyString};
use serde::{Deserialize, Serialize};
use serde_json::Result;
use tfhe::core_crypto::prelude;
use tfhe::core_crypto::prelude::*;

mod compression;
mod computations;
mod encryption;
mod ml;
use pyo3::prelude::*;
// Private Key builder

#[derive(Serialize, Deserialize)]
#[pyclass]
struct PrivateKey {
    inner: prelude::GlweSecretKey<Vec<u32>>,
    post_compression_secret_key: GlweSecretKey<Vec<u32>>,
}

#[pymethods]
impl PrivateKey {
    fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        return Ok(PyBytes::new_bound(py, bincode::serialize(&self).unwrap().as_slice()).into());
    }
    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyBytes>) -> PyResult<PrivateKey> {
        let deserialized: PrivateKey = bincode::deserialize(&content.as_bytes().to_vec()).unwrap();
        return Ok(deserialized);
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass]
struct MatmulCryptoParameters {
    bits_reserved_for_computation: usize,
    glwe_encryption_noise_distribution_stdev: f64,
    encryption_glwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    ciphertext_modulus_bit_count: usize,
}
#[pymethods]
impl MatmulCryptoParameters {
    fn serialize(&self, py: Python) -> PyResult<Py<PyString>> {
        return match serde_json::to_string(&self) {
            Ok(json_str) => Ok(PyString::new_bound(py, &json_str).into()),
            Err(error) => Err(PyValueError::new_err(format!(
                "Can not serialize crypto-parameters {error}"
            ))),
        };
    }
    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyString>) -> PyResult<MatmulCryptoParameters> {
        return match serde_json::from_str(&content.to_str().unwrap()) {
            Ok(p) => Ok(p),
            Err(error) => Err(PyValueError::new_err(format!(
                "Can not deserialize cryptoparameters {error}"
            ))),
        };
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass]
struct CompressionKey {
    inner: compression::CompressionKey<u32>,
}

#[pymethods]
impl CompressionKey {
    fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        return Ok(PyBytes::new_bound(py, bincode::serialize(&self).unwrap().as_slice()).into());
    }
    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyBytes>) -> PyResult<CompressionKey> {
        let deserialized: CompressionKey =
            bincode::deserialize(&content.as_bytes().to_vec()).unwrap();
        return Ok(deserialized);
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass]
struct CipherText {
    inner: crate::ml::SeededCompressedEncryptedVector<u32>,
}

#[pymethods]
impl CipherText {
    fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        return Ok(PyBytes::new_bound(py, bincode::serialize(&self).unwrap().as_slice()).into());
    }
    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyBytes>) -> PyResult<CipherText> {
        let deserialized: CipherText = bincode::deserialize(&content.as_bytes().to_vec()).unwrap();
        return Ok(deserialized);
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass]
struct CompressedResultCipherText {
    inner: Vec<prelude::compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext<u32>>,
}

#[pymethods]
impl CompressedResultCipherText {
    fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        return Ok(PyBytes::new_bound(py, bincode::serialize(&self).unwrap().as_slice()).into());
    }
    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyBytes>) -> PyResult<CompressedResultCipherText> {
        let deserialized: CompressedResultCipherText =
            bincode::deserialize(&content.as_bytes().to_vec()).unwrap();
        return Ok(deserialized);
    }
}

#[pyfunction]
fn create_private_key(
    crypto_params: &MatmulCryptoParameters,
) -> PyResult<(PrivateKey, CompressionKey)> {
    let ciphertext_modulus: CiphertextModulus<u32> =
        CiphertextModulus::try_new_power_of_2(crypto_params.ciphertext_modulus_bit_count).unwrap();
    let mod_switch_bit_count = crypto_params.ciphertext_modulus_bit_count - 1;

    // This could be a method to generate a private key object
    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    let mut secret_rng = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());

    let compression_params = compression::CompressionKeyParameters {
        packing_ks_level: DecompositionLevelCount(2),
        packing_ks_base_log: DecompositionBaseLog(8),
        packing_ks_polynomial_size: PolynomialSize(256),
        packing_ks_glwe_dimension: GlweDimension(5),
        lwe_per_glwe: LweCiphertextCount(256),
        packing_ciphertext_modulus: ciphertext_modulus,
        storage_log_modulus: CiphertextModulusLog(mod_switch_bit_count),
        packing_ks_key_noise_distribution: DynamicDistribution::new_gaussian_from_std_dev(
            StandardDev(
                2.0f64.powi(2) / 2.0f64.powi(crypto_params.ciphertext_modulus_bit_count as i32),
            ),
        ),
    };
    let glwe_secret_key: GlweSecretKey<Vec<u32>> = allocate_and_generate_new_binary_glwe_secret_key(
        crypto_params.encryption_glwe_dimension,
        crypto_params.polynomial_size,
        &mut secret_rng,
    );
    let glwe_secret_key_as_lwe_secret_key = glwe_secret_key.as_lwe_secret_key();
    let (post_compression_glwe_secret_key, compression_key) =
        compression::CompressionKey::new(&glwe_secret_key_as_lwe_secret_key, compression_params);
    return Ok((
        PrivateKey {
            inner: glwe_secret_key,
            post_compression_secret_key: post_compression_glwe_secret_key,
        },
        CompressionKey {
            inner: compression_key,
        },
    ));
}

#[pyfunction]
fn encrypt(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: PyReadonlyArray<u32, Ix1>,
) -> PyResult<CipherText> {
    // Find a better way to use the constants
    // TODO: For now very small noise, find secure noise
    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();

    let mod_switch_bit_count = crypto_params.ciphertext_modulus_bit_count - 1;
    let mod_switch_modulus = CiphertextModulusLog(mod_switch_bit_count);

    let ciphertext_modulus: CiphertextModulus<u32> =
        CiphertextModulus::try_new_power_of_2(crypto_params.ciphertext_modulus_bit_count).unwrap();
    let glwe_encryption_noise_distribution: prelude::DynamicDistribution<u32> =
        DynamicDistribution::new_gaussian_from_std_dev(StandardDev(
            2.0f64.powi(2) / 2.0f64.powi(crypto_params.ciphertext_modulus_bit_count as i32),
        ));
    let seeded_encrypted_vector = ml::SeededCompressedEncryptedVector::new(
        &data.as_array().as_slice().unwrap(),
        &pkey.inner,
        crypto_params.bits_reserved_for_computation,
        mod_switch_modulus,
        glwe_encryption_noise_distribution,
        ciphertext_modulus,
        seeder,
    );
    return Ok(CipherText {
        inner: seeded_encrypted_vector,
    });
}

#[pyfunction]
fn dot_product(
    ciphertext: &CipherText,
    data: PyReadonlyArray<u32, Ix1>,
    compression_key: &CompressionKey,
) -> PyResult<CompressedResultCipherText> {
    let result: crate::ml::EncryptedDotProductResult<u32> = ciphertext
        .inner
        .decompress()
        .dot(data.as_array().as_slice().unwrap());
    let compressed_results = compression_key
        .inner
        .compress_ciphertexts_into_list(&[result]);
    return Ok(CompressedResultCipherText {
        inner: compressed_results,
    });
}

#[pyfunction]
fn decrypt(
    compressed_result: &CompressedResultCipherText,
    private_key: &PrivateKey,
) -> PyResult<u32> {
    let bits_reserved_for_computation = 12;
    let extracted: Vec<_> = compressed_result
        .inner
        .clone()
        .into_iter()
        .map(|compressed| compressed.extract())
        .collect();
    let result = extracted.into_iter().next().unwrap();
    let decrypted_dot: Vec<u32> = encryption::decrypt_glwe(
        &private_key.post_compression_secret_key,
        &result,
        bits_reserved_for_computation,
    );
    let decrypted_dot: u32 = decrypted_dot[0];
    Ok(decrypted_dot)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn concrete_ml_extensions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Maybe we could put this in a loop?
    m.add_function(wrap_pyfunction!(create_private_key, m)?)?;
    m.add_function(wrap_pyfunction!(encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt, m)?)?;
    m.add_class::<CipherText>()?;
    m.add_class::<CompressedResultCipherText>()?;
    m.add_class::<CompressionKey>()?;
    m.add_class::<MatmulCryptoParameters>()?;
    // m.add_class::<PrivateKey>()?;
    Ok(())
}
