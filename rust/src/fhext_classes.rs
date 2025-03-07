#[cfg(all(feature = "python"))]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use tfhe::core_crypto::prelude::*;
use tfhe::core_crypto::prelude;

use crate::Scalar;

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "use_lib2", derive(uniffi::Object))]
#[cfg_attr(feature = "python", pyclass)]
pub struct PrivateKey {
    pub inner: prelude::GlweSecretKey<Vec<Scalar>>,
    pub post_compression_secret_key: GlweSecretKey<Vec<Scalar>>,
}

