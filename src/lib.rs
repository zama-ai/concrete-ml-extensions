#![allow(clippy::excessive_precision)]

// TODO: Implement something like Ix1 dimension handling for GLWECipherTexts

use ml::EncryptedDotProductResult;
use numpy::ndarray::Axis;
use numpy::{Ix1, Ix2, PyArray2, PyArrayMethods, PyReadonlyArray};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyString};
use serde::{Deserialize, Serialize};
use std::ops::Add;
use std::time::{Duration, Instant};
use tfhe::core_crypto::prelude;
use tfhe::core_crypto::prelude::*;

mod compression;
mod computations;
mod encryption;
mod ml;
use pyo3::prelude::*;
use rayon::prelude::*;

// Private Key builder

type Scalar = u64;
#[derive(Serialize, Deserialize, Clone)]
#[pyclass]
pub struct EncryptedMatrix {
    pub inner: Vec<ml::SeededCompressedEncryptedVector<Scalar>>,
    pub shape: (usize, usize),
}

#[derive(Serialize, Deserialize)]
#[pyclass]
struct PrivateKey {
    inner: prelude::GlweSecretKey<Vec<Scalar>>,
    post_compression_secret_key: GlweSecretKey<Vec<Scalar>>,
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
    inner: compression::CompressionKey<Scalar>,
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

#[derive(Serialize, Deserialize, Clone)]
#[pyclass]
struct CipherText {
    inner: crate::ml::SeededCompressedEncryptedVector<Scalar>,
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

#[derive(Serialize, Deserialize, Clone)]
#[pyclass]
struct CompressedResultCipherText {
    inner: Vec<prelude::compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext<Scalar>>,
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

fn internal_encrypt(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: &[Scalar],
) -> Result<CipherText, PyErr> {
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
    Ok(CipherText {
        inner: seeded_encrypted_vector,
    })
}

fn internal_decrypt(
    compressed_result: &CompressedResultCipherText,
    crypto_params: &MatmulCryptoParameters,
    private_key: &PrivateKey,
    num_valid_glwe_values_in_last_ciphertext: usize,
) -> PyResult<Vec<Scalar>> {
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

#[pyfunction]
fn encrypt(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: PyReadonlyArray<Scalar, Ix1>,
) -> PyResult<CipherText> {
    internal_encrypt(pkey, crypto_params, data.as_array().as_slice().unwrap())
}

#[pyfunction]
fn encrypt_matrix(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: PyReadonlyArray<u64, Ix2>,
) -> PyResult<EncryptedMatrix> {
    let mut encrypted_matrix = Vec::new();
    for row in data.as_array().outer_iter() {
        let row_array = row.to_owned();
        let encrypted_row = internal_encrypt(pkey, crypto_params, row_array.as_slice().unwrap())?;
        encrypted_matrix.push(encrypted_row.inner);
    }
    Ok(EncryptedMatrix {
        inner: encrypted_matrix,
        shape: (data.dims()[0], data.dims()[1]),
    })
}

#[pyfunction]
fn matrix_multiplication(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CompressionKey,
) -> PyResult<Vec<CompressedResultCipherText>> {
    let data_array = data.as_array();
    /*
       let data_slice = if let Some(slc) = data_col.as_slice() {
           Array::from_shape_vec(data_col.raw_dim(), slc.to_vec()).unwrap()
       } else {
           Array::from_shape_vec(data_col.raw_dim(), data_col.iter().cloned().collect())
               .unwrap()
       };
    */

    let mut duration_dot = Duration::new(0, 0);
    let mut duration_pks = Duration::new(0, 0);
    let start_0 = Instant::now();

    let data_columns: Vec<_> = data_array
        .axis_iter(Axis(1))
        .map(|col| col.to_owned())
        .collect();

    let result_matrix = encrypted_matrix
        .inner
        .iter()
        .map(|encrypted_row| {
            let now = Instant::now();

            let row_results = data_columns
                .par_iter()
                .map(|data_col| {
                    let data_col_slice = data_col.as_slice().unwrap();
                    internal_dot_product(
                        &CipherText {
                            inner: encrypted_row.clone(),
                        },
                        data_col_slice,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            duration_dot = duration_dot.add(now.elapsed());

            let now = Instant::now();
            let compressed_row = compression_key
                .inner
                .compress_ciphertexts_into_list(&row_results);
            duration_pks = duration_pks.add(now.elapsed());

            Ok(CompressedResultCipherText {
                inner: compressed_row,
            })
        })
        .collect::<Result<Vec<CompressedResultCipherText>, PyErr>>();

    let duration_all = start_0.elapsed();
    println!();
    println!();
    println!(
        "Time in dot {:?}, in pks {:?}, all {:?}",
        duration_dot, duration_pks, duration_all
    );

    Ok(result_matrix?)
}

fn internal_dot_product(
    ciphertext: &CipherText,
    data: &[Scalar],
) -> Result<crate::ml::EncryptedDotProductResult<Scalar>, PyErr> {
    let result: crate::ml::EncryptedDotProductResult<Scalar> =
        ciphertext.inner.decompress().dot(data);
    Ok(result)
}

#[pyfunction]
fn dot_product(
    ciphertext: &CipherText,
    data: PyReadonlyArray<Scalar, Ix1>,
    compression_key: &CompressionKey,
) -> PyResult<CompressedResultCipherText> {
    let result = internal_dot_product(ciphertext, data.as_array().as_slice().unwrap())?;

    let mut result_list = Vec::<EncryptedDotProductResult<Scalar>>::new();
    for _ in 0..compression_key
        .inner
        .packing_key_switching_key
        .output_polynomial_size()
        .0
    {
        result_list.push(result.clone());
    }

    let compressed_results = compression_key
        .inner
        .compress_ciphertexts_into_list(&result_list);

    Ok(CompressedResultCipherText {
        inner: compressed_results,
    })
}

#[pyfunction]
fn decrypt(
    compressed_result: &CompressedResultCipherText,
    private_key: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    num_valid_glwe_values_in_last_ciphertext: usize,
) -> PyResult<Vec<Scalar>> {
    internal_decrypt(
        compressed_result,
        crypto_params,
        private_key,
        num_valid_glwe_values_in_last_ciphertext,
    )
}

#[pyfunction]
fn decrypt_matrix(
    compressed_matrix: Vec<CompressedResultCipherText>,
    private_key: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    num_valid_glwe_values_in_last_ciphertext: usize,
) -> PyResult<Py<PyArray2<Scalar>>> {
    let decrypted_matrix: Vec<Vec<Scalar>> = compressed_matrix
        .iter()
        .map(|compressed_row| {
            internal_decrypt(
                compressed_row,
                crypto_params,
                private_key,
                num_valid_glwe_values_in_last_ciphertext,
            )
        })
        .collect::<Result<_, _>>()?;

    Python::with_gil(|py| {
        let np_array: Bound<'_, PyArray2<Scalar>> =
            PyArray2::from_vec2_bound(py, &decrypted_matrix)?;
        Ok(np_array.into())
    })
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
    m.add_function(wrap_pyfunction!(encrypt_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_multiplication, m)?)?;
    m.add_class::<CipherText>()?;
    m.add_class::<CompressedResultCipherText>()?;
    m.add_class::<CompressionKey>()?;
    m.add_class::<MatmulCryptoParameters>()?;
    m.add_class::<EncryptedMatrix>()?;
    // m.add_class::<PrivateKey>()?;
    Ok(())
}
