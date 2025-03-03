#![allow(clippy::excessive_precision)]

// TODO: Implement something like Ix1 dimension handling for GLWECipherTexts

use numpy::ndarray::Axis;
use numpy::{Ix2, PyArray2, PyArrayMethods, PyReadonlyArray};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyString, PyTuple};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
use tfhe::core_crypto::gpu::is_cuda_available as core_is_cuda_available;
use tfhe::core_crypto::prelude;
use tfhe::core_crypto::prelude::*;
use tfhe::prelude::*;

use tfhe::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;

mod compression;
mod computations;
mod encryption;
mod ml;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::marker::PhantomData;
use tfhe::shortint::ClassicPBSParameters;

use tfhe::{generate_keys, ClientKey, ConfigBuilder, FheInt16, FheInt8, FheUint16, FheUint8};

use tfhe::safe_serialization::{safe_deserialize, safe_serialize};

const BLOCK_PARAMS: ClassicPBSParameters =
    tfhe::shortint::parameters::v0_10::classic::gaussian::p_fail_2_minus_64::ks_pbs::V0_10_PARAM_MESSAGE_2_CARRY_3_KS_PBS_GAUSSIAN_2M64;

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
use tfhe::core_crypto::gpu::entities::lwe_packing_keyswitch_key::CudaLwePackingKeyswitchKey;
#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
use tfhe::core_crypto::gpu::vec::GpuIndex;
#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
use tfhe::core_crypto::gpu::vec::*;
#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
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

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
#[pyclass]
struct CudaCompressionKey {
    inner: compression::CompressionKey<Scalar>,
    buffers: compression::CudaCompressionBuffers<Scalar>,
}

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
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
        let stream = CudaStreams::new_single_gpu(GpuIndex::new(gpu_index));

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

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
#[pyfunction]
fn cuda_create_private_key(
    crypto_params: &MatmulCryptoParameters,
) -> PyResult<(PrivateKey, CudaCompressionKey)> {
    let (glwe_secret_key, post_compression_glwe_secret_key, compression_key) =
        create_private_key_internal(crypto_params);

    let gpu_index = 0;
    let stream = CudaStreams::new_single_gpu(GpuIndex::new(gpu_index));
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

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
#[pyfunction]
fn cuda_matrix_multiplication(
    encrypted_matrix: &EncryptedMatrix,
    data: &CudaClearMatrix,
    compression_key: &CudaCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    let poly_size_in = encrypted_matrix.inner[0].data[0].polynomial_size().0;
    assert_eq!(
        poly_size_in, compression_key.inner.lwe_per_glwe.0,
        "GPU GLWE dot product only supports crypto-params
         where the output lwe/glwe count is equal to the input poly size"
    );

    let gpu_index = 0;
    let stream = CudaStreams::new_single_gpu(GpuIndex::new(gpu_index));

    let ciphertext_modulus = compression_key
        .inner
        .packing_key_switching_key
        .ciphertext_modulus();

    let mut d_accum_buffers = Vec::<CudaLweCiphertextList<Scalar>>::with_capacity(data.col_blocks);
    let mut d_instant_buffers =
        Vec::<CudaLweCiphertextList<Scalar>>::with_capacity(data.col_blocks);

    (0..data.col_blocks).for_each(|j| {
        let lwe_count = if j == data.col_blocks - 1 && data.col_total % poly_size_in > 0 {
            data.col_total % poly_size_in
        } else {
            poly_size_in
        };

        let h_output_lwe = LweCiphertextList::new(
            Scalar::ZERO,
            LweSize(poly_size_in + 1),
            LweCiphertextCount(lwe_count),
            ciphertext_modulus,
        );

        let d_accum_output_lwe: CudaLweCiphertextList<u64> =
            CudaLweCiphertextList::from_lwe_ciphertext_list(&h_output_lwe, &stream);

        let d_output_lwe: CudaLweCiphertextList<u64> =
            CudaLweCiphertextList::from_lwe_ciphertext_list(&h_output_lwe, &stream);

        d_accum_buffers.push(d_accum_output_lwe);

        d_instant_buffers.push(d_output_lwe);
    });

    // GPU polynomial product
    let result_matrix = encrypted_matrix
        .inner
        .iter()
        .map(|encrypted_row| {
            let decompressed_row = encrypted_row.decompress();

            let compressed_row = (0..data.col_blocks).map(|j| {
                let lwe_count = if j == data.col_blocks - 1 && data.col_total % poly_size_in > 0 {
                    data.col_total % poly_size_in
                } else {
                    poly_size_in
                };

                let mut d_accum_output_lwe = d_accum_buffers.get_mut(j).unwrap();
                let mut d_output_lwe = d_instant_buffers.get_mut(j).unwrap();


                unsafe {
                    d_accum_output_lwe.set_to_zero_async(&stream);
                }

                for i in 0..data.row_blocks {
                    let h_data_block = &data.data[j][i];
                    assert_eq!(h_data_block.len(), lwe_count * poly_size_in, "Cached GPU matrix block has wrong size");

                    decompressed_row.cuda_accum_dot_with_clear_matrix_block(i, &h_data_block, &mut d_accum_output_lwe, &mut d_output_lwe, &stream);
                }

                let compresed_chunk: compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext<u64> = compression_key
                    .inner
                    .cuda_compress_ciphertexts_into_single_glwe(&d_accum_output_lwe, &compression_key.buffers);

                compresed_chunk
            }).collect();

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
            let decompressed_row = encrypted_row.decompress();

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

// NEW PARAMS: LEVELS==1 for speed
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

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
#[pyfunction]
fn is_cuda_available() -> PyResult<bool> {
    return Ok(core_is_cuda_available());
}

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
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

    m.add_function(wrap_pyfunction!(encrypt_serialize_u8_radix_2d, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_serialized_u8_radix_2d, m)?)?;

    m.add_function(wrap_pyfunction!(encrypt_serialize_i8_radix_2d, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_serialized_i8_radix_2d, m)?)?;

    m.add_function(wrap_pyfunction!(decrypt_serialized_u16_radix_2d, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_serialized_i16_radix_2d, m)?)?;

    m.add_function(wrap_pyfunction!(keygen_radix, m)?)?;
    m.add_function(wrap_pyfunction!(get_crypto_params_radix, m)?)?;
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
#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
#[pymodule]
fn concrete_ml_extensions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cuda_create_private_key, m)?)?;
    m.add_function(wrap_pyfunction!(is_cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_matrix_multiplication, m)?)?;
    m.add_function(wrap_pyfunction!(make_cuda_clear_matrix, m)?)?;

    m.add_class::<CudaCompressionKey>()?;
    m.add_class::<CudaClearMatrix>()?;

    concrete_ml_extensions_base(m)
}

#[cfg(not(feature = "cuda"))]
#[pymodule]
fn concrete_ml_extensions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    concrete_ml_extensions_base(m)
}

const SERIALIZE_SIZE_LIMIT: u64 = 1_000_000_000;

#[pyfunction]
fn encrypt_serialize_u8_radix_2d(
    py: Python,
    value: PyReadonlyArray<u8, Ix2>,
    client_key_ser: Py<PyBytes>,
) -> PyResult<Py<PyBytes>> {
    let arr = value.as_array();

    let client_key: ClientKey =
        safe_deserialize(client_key_ser.as_bytes(py), SERIALIZE_SIZE_LIMIT).unwrap();

    let data_vec = arr.as_standard_layout().into_owned().into_raw_vec();

    let cts: Vec<FheUint8> = data_vec
        .iter()
        .map(|v| FheUint8::encrypt(v.clone(), &client_key))
        .collect();

    let serialized = bincode::serialize(cts.as_slice()).unwrap();

    Ok(PyBytes::new_bound(py, serialized.as_slice()).into())
}

#[pyfunction]
fn encrypt_serialize_i8_radix_2d(
    py: Python,
    value: PyReadonlyArray<i8, Ix2>,
    client_key_ser: Py<PyBytes>,
) -> PyResult<Py<PyBytes>> {
    let arr = value.as_array();

    let client_key: ClientKey =
        safe_deserialize(client_key_ser.as_bytes(py), SERIALIZE_SIZE_LIMIT).unwrap();

    let data_vec = arr.as_standard_layout().into_owned().into_raw_vec();

    let cts: Vec<FheInt8> = data_vec
        .iter()
        .map(|v| FheInt8::encrypt(v.clone(), &client_key))
        .collect();

    let serialized = bincode::serialize(cts.as_slice()).unwrap();

    Ok(PyBytes::new_bound(py, serialized.as_slice()).into())
}

#[pyfunction]
fn decrypt_serialized_u8_radix_2d(
    py: Python,
    value: Py<PyBytes>,
    num_cols: usize,
    client_key_ser: Py<PyBytes>,
) -> PyResult<Py<PyArray2<u8>>> {
    let client_key: ClientKey =
        safe_deserialize(client_key_ser.as_bytes(py), SERIALIZE_SIZE_LIMIT).unwrap();

    let fheint_array: Vec<FheUint8> = bincode::deserialize(value.as_bytes(py)).unwrap();

    let results: Vec<u8> = fheint_array
        .iter()
        .map(|v| v.decrypt(&client_key))
        .collect();

    let results2d: Vec<Vec<u8>> = results
        .into_chunks(num_cols)
        .map(|sl| sl.to_vec())
        .collect();

    Python::with_gil(|py| {
        let np_array: Bound<'_, PyArray2<u8>> = PyArray2::from_vec2_bound(py, &results2d)?;
        Ok(np_array.into())
    })
}

#[pyfunction]
fn decrypt_serialized_u16_radix_2d(
    py: Python,
    value: Py<PyBytes>,
    num_cols: usize,
    client_key_ser: Py<PyBytes>,
) -> PyResult<Py<PyArray2<u16>>> {
    let client_key: ClientKey =
        safe_deserialize(client_key_ser.as_bytes(py), SERIALIZE_SIZE_LIMIT).unwrap();

    let fheint_array: Vec<FheUint16> = bincode::deserialize(value.as_bytes(py)).unwrap();

    let results: Vec<u16> = fheint_array
        .iter()
        .map(|v| v.decrypt(&client_key))
        .collect();

    let results2d: Vec<Vec<u16>> = results
        .into_chunks(num_cols)
        .map(|sl| sl.to_vec())
        .collect();

    Python::with_gil(|py| {
        let np_array: Bound<'_, PyArray2<u16>> = PyArray2::from_vec2_bound(py, &results2d)?;
        Ok(np_array.into())
    })
}

#[pyfunction]
fn decrypt_serialized_i16_radix_2d(
    py: Python,
    value: Py<PyBytes>,
    num_cols: usize,
    client_key_ser: Py<PyBytes>,
) -> PyResult<Py<PyArray2<i16>>> {
    let client_key: ClientKey =
        safe_deserialize(client_key_ser.as_bytes(py), SERIALIZE_SIZE_LIMIT).unwrap();

    let fheint_array: Vec<FheInt16> = bincode::deserialize(value.as_bytes(py)).unwrap();

    let results: Vec<i16> = fheint_array
        .iter()
        .map(|v| v.decrypt(&client_key))
        .collect();

    let results2d: Vec<Vec<i16>> = results
        .into_chunks(num_cols)
        .map(|sl| sl.to_vec())
        .collect();

    Python::with_gil(|py| {
        let np_array: Bound<'_, PyArray2<i16>> = PyArray2::from_vec2_bound(py, &results2d)?;
        Ok(np_array.into())
    })
}

#[pyfunction]
fn decrypt_serialized_i8_radix_2d(
    py: Python,
    value: Py<PyBytes>,
    num_cols: usize,
    client_key_ser: Py<PyBytes>,
) -> PyResult<Py<PyArray2<i8>>> {
    let client_key: ClientKey =
        safe_deserialize(client_key_ser.as_bytes(py), SERIALIZE_SIZE_LIMIT).unwrap();

    let fheint_array: Vec<FheInt8> = bincode::deserialize(value.as_bytes(py)).unwrap();

    let results: Vec<i8> = fheint_array
        .iter()
        .map(|v| v.decrypt(&client_key))
        .collect();

    let results2d: Vec<Vec<i8>> = results
        .into_chunks(num_cols)
        .map(|sl| sl.to_vec())
        .collect();

    Python::with_gil(|py| {
        let np_array: Bound<'_, PyArray2<i8>> = PyArray2::from_vec2_bound(py, &results2d)?;
        Ok(np_array.into())
    })
}

#[pyfunction]
fn keygen_radix(py: Python<'_>) -> PyResult<Bound<PyTuple>> {
    let config = ConfigBuilder::with_custom_parameters(BLOCK_PARAMS).build();

    let (client_key, server_key) = generate_keys(config);
    let (integer_ck, _, _, _) = client_key.clone().into_raw_parts();
    let shortint_ck = integer_ck.into_raw_parts();
    assert!(BLOCK_PARAMS.encryption_key_choice == EncryptionKeyChoice::Big);
    let (glwe_secret_key, _, _) = shortint_ck.into_raw_parts();
    let lwe_secret_key = glwe_secret_key.into_lwe_secret_key();

    let mut ck_ser: Vec<u8> = vec![];
    let _ = safe_serialize(&client_key, &mut ck_ser, SERIALIZE_SIZE_LIMIT);

    let mut bsk_ser: Vec<u8> = vec![];
    let _ = safe_serialize(&server_key, &mut bsk_ser, SERIALIZE_SIZE_LIMIT);

    let mut lwe_ck_ser: Vec<u8> = vec![];
    let _ = safe_serialize(&lwe_secret_key, &mut lwe_ck_ser, SERIALIZE_SIZE_LIMIT);

    Ok(PyTuple::new_bound(
        py,
        vec![
            PyBytes::new_bound(py, ck_ser.as_slice()),
            PyBytes::new_bound(py, bsk_ser.as_slice()),
            PyBytes::new_bound(py, lwe_ck_ser.as_slice()),
        ],
    ))
}

#[pyfunction]
fn get_crypto_params_radix() -> String {
    serde_json::to_string(&BLOCK_PARAMS).unwrap()
}

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
#[pyclass]
struct CudaClearMatrix {
    data: Vec<Vec<CudaVec<Scalar>>>, // stores the GPU buffers for each block
    row_blocks: usize,               // number of row blocks (of size=poly_size)
    row_total: usize,                // total number of rows
    col_blocks: usize,               // number of column blocks (of size=poly_size)
    col_total: usize,                // total number of columns
}

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
#[pyfunction]
fn make_cuda_clear_matrix(
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CudaCompressionKey,
) -> CudaClearMatrix {
    let poly_size_in = compression_key.inner.lwe_per_glwe.0;
    let data_array = data.as_array();

    let n_blocks_rows = (data_array.nrows() + poly_size_in - 1) / poly_size_in;
    let n_blocks_cols = (data_array.ncols() + poly_size_in - 1) / poly_size_in;

    let gpu_index = 0;
    let streams = CudaStreams::new_single_gpu(GpuIndex::new(gpu_index));

    let mut gpu_buf_matrix = Vec::<Vec<CudaVec<Scalar>>>::new();
    for j in 0..n_blocks_cols {
        let mut gpu_buf_row = Vec::<CudaVec<Scalar>>::new();

        let j0 = j * poly_size_in;
        let j1 = if j == n_blocks_cols - 1 {
            data_array.ncols()
        } else {
            (j + 1) * poly_size_in
        };

        for i in 0..n_blocks_rows {
            let i0 = i * poly_size_in;
            let i1 = if i == n_blocks_rows - 1 {
                data_array.nrows()
            } else {
                (i + 1) * poly_size_in
            };

            let block = data_array.slice(numpy::ndarray::s![i0..i1, j0..j1]);

            let mut data_block = Vec::<Scalar>::with_capacity(block.ncols() * poly_size_in);

            block.axis_iter(Axis(1)).for_each(|col| {
                let mut reversed: Vec<Scalar> = col.to_vec();
                reversed
                    .extend(std::iter::repeat(Scalar::ZERO).take(poly_size_in - reversed.len()));
                reversed.reverse();
                data_block.append(&mut reversed);
            });

            unsafe {
                let d_clear_matrix = CudaVec::from_cpu_async(data_block.as_ref(), &streams, 0);

                gpu_buf_row.push(d_clear_matrix);
            }
        }
        gpu_buf_matrix.push(gpu_buf_row);
    }

    CudaClearMatrix {
        data: gpu_buf_matrix,
        row_blocks: n_blocks_rows,
        col_blocks: n_blocks_cols,
        row_total: data_array.nrows(),
        col_total: data_array.ncols(),
    }
}
