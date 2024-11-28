#![allow(clippy::excessive_precision)]

// TODO: Implement something like Ix1 dimension handling for GLWECipherTexts

use numpy::ndarray::Axis;
use numpy::{Ix2, PyArray2, PyArrayMethods, PyReadonlyArray};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyString};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use tfhe::core_crypto::gpu::is_cuda_available as core_is_cuda_available;
use tfhe::core_crypto::prelude;
use tfhe::core_crypto::prelude::*;
mod compression;
mod computations;
mod encryption;
mod ml;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::marker::PhantomData;

#[cfg(feature = "cuda")]
use tfhe::core_crypto::gpu::entities::lwe_packing_keyswitch_key::CudaLwePackingKeyswitchKey;
#[cfg(feature = "cuda")]
use tfhe::core_crypto::gpu::CudaStreams;

// Private Key builder

//use std::time::Instant;

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
struct CpuCompressionKey {
    inner: compression::CompressionKey<Scalar>,
    buffers: compression::CpuCompressionBuffers<Scalar>,
}

#[pymethods]
impl CpuCompressionKey {
    fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        return Ok(
            PyBytes::new_bound(py, bincode::serialize(&self.inner).unwrap().as_slice()).into(),
        );
    }

    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyBytes>) -> PyResult<CpuCompressionKey> {
        let deserialized: CpuCompressionKey =
            bincode::deserialize(&content.as_bytes().to_vec()).unwrap();

        return Ok(deserialized);
    }
}

#[cfg(feature = "cuda")]
#[pyclass]
struct CudaCompressionKey {
    inner: compression::CompressionKey<Scalar>,
    buffers: compression::CudaCompressionBuffers<Scalar>,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl CudaCompressionKey {
    fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        return Ok(
            PyBytes::new_bound(py, bincode::serialize(&self.inner).unwrap().as_slice()).into(),
        );
    }

    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyBytes>) -> PyResult<CudaCompressionKey> {
        let gpu_index = 0;
        let stream = CudaStreams::new_single_gpu(gpu_index);

        let deserialized: compression::CompressionKey<Scalar> =
            bincode::deserialize(&content.as_bytes().to_vec()).unwrap();

        let cuda_pksk = CudaLwePackingKeyswitchKey::from_lwe_packing_keyswitch_key(
            &deserialized.packing_key_switching_key,
            &stream,
        );

        return Ok(CudaCompressionKey {
            inner: deserialized,
            buffers: compression::CudaCompressionBuffers {
                cuda_pksk: cuda_pksk,
            },
        });
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

#[pymethods]
impl EncryptedMatrix {
    fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        return Ok(PyBytes::new_bound(py, bincode::serialize(&self).unwrap().as_slice()).into());
    }
    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyBytes>) -> PyResult<EncryptedMatrix> {
        let deserialized: EncryptedMatrix =
            bincode::deserialize(&content.as_bytes().to_vec()).unwrap();
        return Ok(deserialized);
    }
}

#[derive(Serialize, Deserialize, Clone)]
//#[pyclass]
struct CompressedResultCipherText {
    inner: Vec<prelude::compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext<Scalar>>,
}

#[derive(Serialize, Deserialize, Clone)]
#[pyclass]
struct CompressedResultEncryptedMatrix {
    inner: Vec<CompressedResultCipherText>,
}

#[pymethods]
impl CompressedResultEncryptedMatrix {
    fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        return Ok(PyBytes::new_bound(py, bincode::serialize(&self).unwrap().as_slice()).into());
    }
    #[staticmethod]
    fn deserialize(content: &Bound<'_, PyBytes>) -> PyResult<CompressedResultEncryptedMatrix> {
        let deserialized: CompressedResultEncryptedMatrix =
            bincode::deserialize(&content.as_bytes().to_vec()).unwrap();
        return Ok(deserialized);
    }
}

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

#[pyfunction]
fn cpu_create_private_key(
    crypto_params: &MatmulCryptoParameters,
) -> PyResult<(PrivateKey, CpuCompressionKey)> {
    let (glwe_secret_key, post_compression_glwe_secret_key, compression_key) =
        create_private_key_internal(crypto_params);

    return Ok((
        PrivateKey {
            inner: glwe_secret_key,
            post_compression_secret_key: post_compression_glwe_secret_key,
        },
        CpuCompressionKey {
            inner: compression_key,
            buffers: compression::CpuCompressionBuffers::<Scalar> { _tmp: PhantomData },
        },
    ));
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn cuda_create_private_key(
    crypto_params: &MatmulCryptoParameters,
) -> PyResult<(PrivateKey, CudaCompressionKey)> {
    let (glwe_secret_key, post_compression_glwe_secret_key, compression_key) =
        create_private_key_internal(crypto_params);

    let gpu_index = 0;
    let stream = CudaStreams::new_single_gpu(gpu_index);
    let cuda_pksk = CudaLwePackingKeyswitchKey::from_lwe_packing_keyswitch_key(
        &compression_key.packing_key_switching_key,
        &stream,
    );

    return Ok((
        PrivateKey {
            inner: glwe_secret_key,
            post_compression_secret_key: post_compression_glwe_secret_key,
        },
        CudaCompressionKey {
            inner: compression_key,
            buffers: compression::CudaCompressionBuffers {
                cuda_pksk: cuda_pksk,
            },
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
fn encrypt_matrix(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: PyReadonlyArray<Scalar, Ix2>,
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

#[cfg(feature = "cuda")]
#[pyfunction]
fn cuda_matrix_multiplication(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CudaCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    let data_array = data.as_array();

    let data_columns: Vec<_> = data_array
        .axis_iter(Axis(1))
        .map(|col| col.to_owned())
        .collect();

    let poly_size_compress = compression_key
        .inner
        .packing_key_switching_key
        .output_polynomial_size();
    let lwe_size = LweSize(poly_size_compress.0 + 1);
    let lwe_count = LweCiphertextCount(data_columns.len());

    let mut row_results2 = LweCiphertextList::new(
        0,
        lwe_size,
        lwe_count,
        compression_key
            .inner
            .packing_key_switching_key
            .ciphertext_modulus(),
    );

    let result_matrix = encrypted_matrix
        .inner
        .iter()
        .map(|encrypted_row| {
//            let now = Instant::now();
            let decompressed_row = encrypted_row.decompress();
            //            println!("DECOMPRESS : {}ms", now.elapsed().as_millis());

            //            let now = Instant::now();

            data_columns
                .par_iter()
                .zip(row_results2.par_iter_mut())
                .for_each(|(data_col, mut result_out)| {
                    let data_col_slice = data_col.as_slice().unwrap();
                    result_out.as_mut().copy_from_slice(
                        decompressed_row
                            .dot(data_col_slice)
                            .as_lwe()
                            .into_container(),
                    );
                });

            //            println!("POLY MUL TIME : {}ms", now.elapsed().as_millis());

            let compressed_row = compression_key
                .inner
                .cuda_compress_ciphertexts_into_list(&row_results2, &compression_key.buffers);

            Ok(CompressedResultCipherText {
                inner: compressed_row,
            })
        })
        .collect::<Result<Vec<CompressedResultCipherText>, PyErr>>();

    Ok(CompressedResultEncryptedMatrix {
        inner: result_matrix?,
    })
}

#[pyfunction]
fn cpu_matrix_multiplication(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    let data_array = data.as_array();

    let data_columns: Vec<_> = data_array
        .axis_iter(Axis(1))
        .map(|col| col.to_owned())
        .collect();

    let poly_size_compress = compression_key
        .inner
        .packing_key_switching_key
        .output_polynomial_size();
    let lwe_size = LweSize(poly_size_compress.0 + 1);
    let lwe_count = LweCiphertextCount(data_columns.len());

    let mut row_results2 = LweCiphertextList::new(
        0,
        lwe_size,
        lwe_count,
        compression_key
            .inner
            .packing_key_switching_key
            .ciphertext_modulus(),
    );

    let result_matrix = encrypted_matrix
        .inner
        .iter()
        .map(|encrypted_row| {
//            let now = Instant::now();
            let decompressed_row = encrypted_row.decompress();
            //            println!("DECOMPRESS : {}ms", now.elapsed().as_millis());

            //            let now = Instant::now();

            data_columns
                .par_iter()
                .zip(row_results2.par_iter_mut())
                .for_each(|(data_col, mut result_out)| {
                    let data_col_slice = data_col.as_slice().unwrap();
                    result_out.as_mut().copy_from_slice(
                        decompressed_row
                            .dot(data_col_slice)
                            .as_lwe()
                            .into_container(),
                    );
                });

            //            println!("POLY MUL TIME : {}ms", now.elapsed().as_millis());

            let compressed_row = compression_key
                .inner
                .cpu_compress_ciphertexts_into_list(&row_results2, &compression_key.buffers);

            Ok(CompressedResultCipherText {
                inner: compressed_row,
            })
        })
        .collect::<Result<Vec<CompressedResultCipherText>, PyErr>>();

    Ok(CompressedResultEncryptedMatrix {
        inner: result_matrix?,
    })
}

#[pyfunction]
fn decrypt_matrix(
    compressed_matrix: CompressedResultEncryptedMatrix,
    private_key: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    num_valid_glwe_values_in_last_ciphertext: usize,
) -> PyResult<Py<PyArray2<Scalar>>> {
    let decrypted_matrix: Vec<Vec<Scalar>> = compressed_matrix
        .inner
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

#[pyfunction]
fn default_params() -> String {
    PARAMS_8B_2048_NEW.to_string()
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn is_cuda_available() -> PyResult<bool> {
    return Ok(core_is_cuda_available());
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn is_cuda_enabled() -> PyResult<bool> {
    return Ok(true);
}
#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn is_cuda_enabled() -> PyResult<bool> {
    return Ok(false);
}

fn concrete_ml_extensions_base(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Maybe we could put this in a loop?
    m.add_function(wrap_pyfunction!(cpu_create_private_key, m)?)?;
    m.add_function(wrap_pyfunction!(encrypt_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_matrix_multiplication, m)?)?;
    m.add_function(wrap_pyfunction!(default_params, m)?)?;
    m.add_function(wrap_pyfunction!(is_cuda_enabled, m)?)?;
    m.add_class::<CipherText>()?;
    m.add_class::<CompressedResultEncryptedMatrix>()?;
    m.add_class::<CpuCompressionKey>()?;
    m.add_class::<MatmulCryptoParameters>()?;
    m.add_class::<EncryptedMatrix>()?;
    // m.add_class::<PrivateKey>()?;
    Ok(())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[cfg(feature = "cuda")]
#[pymodule]
fn concrete_ml_extensions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cuda_create_private_key, m)?)?;
    m.add_function(wrap_pyfunction!(is_cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_matrix_multiplication, m)?)?;
    m.add_class::<CudaCompressionKey>()?;

    concrete_ml_extensions_base(m)
}

#[cfg(not(feature = "cuda"))]
#[pymodule]
fn concrete_ml_extensions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    concrete_ml_extensions_base(m)
}
