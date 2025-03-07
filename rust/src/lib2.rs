#![allow(clippy::excessive_precision)]

use serde::{Deserialize, Serialize};
use tfhe::core_crypto::prelude;
use tfhe::core_crypto::prelude::*;

use crate::compression;
use crate::encryption;
use crate::ml;

use crate::fhext_classes::*;

type Scalar = u64;
uniffi::setup_scaffolding!();

// ===== Custom Error =====

#[derive(Debug, uniffi::Error)]
pub enum MyError {
    GenericError(String),
    SerializationFailed(String),
    DeserializationFailed(String),
}

use std::fmt;
impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MyError::GenericError(msg) => write!(f, "Error: {}", msg),
            MyError::SerializationFailed(msg) => write!(f, "Serialization failed: {}", msg),
            MyError::DeserializationFailed(msg) => write!(f, "Deserialization failed: {}", msg),
        }
    }
}



// ===== Private Key =====



#[uniffi::export]
impl PrivateKey {
    /// Serialize into a byte vector.
    fn serialize(&self) -> Result<Vec<u8>, MyError> {
        bincode::serialize(&self)
            .map_err(|e| MyError::SerializationFailed(e.to_string()))
    }
}

/// Deserialize from a byte vector into a `PrivateKey` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn PrivateKey_deserialize(content: Vec<u8>) -> PrivateKey {
    bincode::deserialize(&content).unwrap()
}

#[uniffi::export]
impl MatmulCryptoParameters {
    /// Serialize into a JSON string.
    fn serialize(&self) -> Result<String, MyError> {
        serde_json::to_string(&self)
            .map_err(|e| MyError::SerializationFailed(e.to_string()))
    }
}

/// Deserialize from a JSON string into a `MatmulCryptoParameters` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn MatmulCryptoParameters_deserialize(content: String) -> Result<MatmulCryptoParameters, MyError> {
    serde_json::from_str(&content)
        .map_err(|e| MyError::DeserializationFailed(e.to_string()))
}



// ===== CpuCompressionKey =====

#[uniffi::export]
impl CpuCompressionKey {
    /// Serialize into a byte vector.
    fn serialize(&self) -> Result<Vec<u8>, MyError> {
        bincode::serialize(&self.inner)
            .map_err(|e| MyError::SerializationFailed(e.to_string()))
    }
}

/// Deserialize from a byte vector into a `CpuCompressionKey` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn CpuCompressionKey_deserialize(content: Vec<u8>) -> Result<CpuCompressionKey, MyError> {
    bincode::deserialize(&content)
        .map_err(|e| MyError::DeserializationFailed(e.to_string()))
}



// ===== CipherText =====



#[uniffi::export]
impl CipherText {
    /// Serialize into a byte vector.
    fn serialize(&self) -> Result<Vec<u8>, MyError> {
        bincode::serialize(&self)
            .map_err(|e| MyError::SerializationFailed(e.to_string()))
    }
}

/// Deserialize from a byte vector into a `CipherText` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn CipherText_deserialize(content: Vec<u8>) -> Result<CipherText, MyError> {
    bincode::deserialize(&content)
        .map_err(|e| MyError::DeserializationFailed(e.to_string()))
}



// ===== EncryptedMatrix =====

#[uniffi::export]
impl EncryptedMatrix {
    fn serialize(&self) -> Result<Vec<u8>, MyError> {
        bincode::serialize(&self)
            .map_err(|e| MyError::SerializationFailed(e.to_string()))
    }
}

/// Deserialize from a byte vector into an `EncryptedMatrix` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn EncryptedMatrix_deserialize(content: Vec<u8>) -> Result<CipherText, MyError> {
    bincode::deserialize(&content)
        .map_err(|e| MyError::DeserializationFailed(e.to_string()))
}



// ===== CompressedResultCipherText =====





// ===== CompressedResultEncryptedMatrix =====


#[uniffi::export]
impl CompressedResultEncryptedMatrix {
    fn serialize(&self) -> Result<Vec<u8>, MyError> {
        bincode::serialize(&self)
            .map_err(|e| MyError::SerializationFailed(e.to_string()))
    }
}

/// Deserialize from a byte vector into an `CompressedResultEncryptedMatrix` object.
/// Must be a standalone function because UniFFI does not support static methods.
#[uniffi::export]
#[allow(non_snake_case)]
fn CompressedResultEncryptedMatrix_deserialize(content: Vec<u8>) -> Result<CompressedResultEncryptedMatrix, MyError> {
    bincode::deserialize(&content)
        .map_err(|e| MyError::DeserializationFailed(e.to_string()))
}



// ===== Key Gen =====

// UniFFI does not support tuples, so we use this struct instead.
#[derive(uniffi::Object)]
struct CPUCreateKeysResult {
    private_key: PrivateKey,
    cpu_compression_key: CpuCompressionKey,
}

#[uniffi::export]
impl CPUCreateKeysResult {
    pub fn private_key(&self) -> PrivateKey {
        self.private_key.clone()
    }

    pub fn cpu_compression_key(&self) -> CpuCompressionKey {
        self.cpu_compression_key.clone()
    }
}

#[uniffi::export]
fn cpu_create_private_key(
    crypto_params: &MatmulCryptoParameters,
) -> CPUCreateKeysResult {
    let (glwe_secret_key, post_compression_glwe_secret_key, compression_key) =
        create_private_key_internal(crypto_params);

    return CPUCreateKeysResult {
        private_key: PrivateKey {
            inner: glwe_secret_key,
            post_compression_secret_key: post_compression_glwe_secret_key,
        },
        cpu_compression_key: CpuCompressionKey {
            inner: compression_key,
        },
    };
}

#[uniffi::export]
fn encrypt_matrix(
    pkey: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    data: Vec<Vec<Scalar>>,
) -> Result<EncryptedMatrix, MyError> {
    let mut encrypted_matrix = Vec::new();
    for row in data.iter() {
        let row_array = row.clone();
        let encrypted_row = internal_encrypt(pkey, crypto_params, row_array.as_slice());
        encrypted_matrix.push(encrypted_row.inner);
    }
    
    Ok(EncryptedMatrix {
        inner: encrypted_matrix,
        shape: (data.len(), data[0].len()),
    })
}

#[uniffi::export]
fn decrypt_matrix(
    compressed_matrix: &CompressedResultEncryptedMatrix, // Changed compressed_matrix to be borrowed
    private_key: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    num_valid_glwe_values_in_last_ciphertext: u64, // Changed `usize` to `u64`
) -> Result<Vec<Vec<Scalar>>, MyError> {
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
        .collect::<Vec<_>>(); 
        //.map_err(|e| MyError::GenericError(format!("Decryption failed: {}", e)))?;

    Ok(decrypted_matrix)
}

#[uniffi::export]
fn default_params() -> String {
    PARAMS_8B_2048_NEW.to_string()
}

/*
// Notable changes made to support UniFFI:

1. `cpu_create_private_key()`: Changed to return a struct rather than a tuple. Also, made both keys 'Clone'
Reason: UniFFI does not support tuples.
```
fn cpu_create_private_key(crypto_params: &MatmulCryptoParameters) -> CPUCreateKeysResult
```

2. `decrypt_matrix()`: Made matrix argument borrow instead of ownership.
Reason: Workaround error `the trait `Lift<lib2::UniFfiTag>` is not implemented for `CompressedResultEncryptedMatrix``

3. `decrypt_matrix()`: Made last param u64 instead of usize
Reason: UniFFI Recommendation: Avoid usize in UniFFI-exposed functions, convert usize to u64/u32 when passing values across FFI boundaries. Use .as usize inside Rust code if needed.
```
fn decrypt_matrix(
    compressed_matrix: &CompressedResultEncryptedMatrix, // Changed compressed_matrix to be borrowed
    private_key: &PrivateKey,
    crypto_params: &MatmulCryptoParameters,
    num_valid_glwe_values_in_last_ciphertext: u64, // Changed `usize` to `u64`
) -> Result<Vec<Vec<Scalar>>, MyError> {
```

4. Made several `deserialize` static methods free-floating functions instead, prefixed with type name.
Reason: UniFFI doesn't support static functions.
```
fn PrivateKey_deserialize(content: Vec<u8>) -> PrivateKey {
  bincode::deserialize(&content).unwrap()
}
```

5. Introduced custom Error type
Reason: UniFFI does not support returning Result<T, String>.
```
pub enum MyError {
    GenericError(String),
    SerializationFailed(String),
    DeserializationFailed(String),
}

use std::fmt;
impl fmt::Display for MyError { … }
```
*/
