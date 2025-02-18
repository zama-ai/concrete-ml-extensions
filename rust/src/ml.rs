use tfhe::core_crypto::commons::math::random::{Distribution, Uniform};
use tfhe::core_crypto::gpu::glwe_ciphertext_list::CudaGlweCiphertextList;
use tfhe::core_crypto::gpu::glwe_linear_algebra::cuda_glwe_dot_product_with_clear_one_to_many;
use tfhe::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use tfhe::core_crypto::gpu::vec::CudaVec;
use tfhe::core_crypto::gpu::{cuda_lwe_ciphertext_add_assign, cuda_lwe_ciphertext_add_assign_async, cuda_lwe_ciphertext_add_async, CudaStreams};
use tfhe::core_crypto::prelude::*;

#[derive(Clone)]
pub struct EncryptedDotProductResult<Scalar: UnsignedInteger> {
    data: LweCiphertextOwned<Scalar>,
}

impl<Scalar: UnsignedInteger> EncryptedDotProductResult<Scalar> {
    pub fn as_lwe(&self) -> LweCiphertextView<'_, Scalar> {
        self.data.as_view()
    }
}

impl<Scalar: UnsignedTorus> EncryptedDotProductResult<Scalar> {
    pub fn decrypt<KeyCont: Container<Element = Scalar>>(
        &self,
        lwe_secret_key: &LweSecretKey<KeyCont>,
        bits_reserved_for_computation: usize,
    ) -> Scalar {
        crate::encryption::decrypt_lwe(lwe_secret_key, &self.data, bits_reserved_for_computation)
    }
}

pub struct EncryptedVector<Scalar: UnsignedInteger> {
    data: Vec<GlweCiphertextOwned<Scalar>>,
    actual_len: usize,
}

impl<Scalar: UnsignedTorus> EncryptedVector<Scalar> {
    pub fn decrypt<KeyCont>(
        &self,
        glwe_secret_key: &GlweSecretKey<KeyCont>,
        bits_reserved_for_computation: usize,
    ) -> Vec<Scalar>
    where
        KeyCont: Container<Element = Scalar>,
    {
        let mut vec: Vec<_> = self
            .data
            .iter()
            .flat_map(|glwe| {
                crate::encryption::decrypt_glwe(
                    glwe_secret_key,
                    glwe,
                    bits_reserved_for_computation,
                )
            })
            .collect();

        vec.resize(self.actual_len, Scalar::ZERO);

        vec
    }

    pub fn cuda_accum_dot_with_clear_matrix_block(&self, 
        glwe_index: usize, 
        other_columns_block: &[Scalar], 
        d_accum_lwe: &mut CudaLweCiphertextList<Scalar>, 
        d_output_lwe: &mut CudaLweCiphertextList<Scalar>, 
        streams: &CudaStreams)
    {
        unsafe {
            let d_clear_matrix = CudaVec::from_cpu_async(other_columns_block.as_ref(), &streams, 0);

            let glwe = self.data.get(glwe_index).unwrap();
            let d_input_glwe = CudaGlweCiphertextList::from_glwe_ciphertext(&glwe, &streams);
            
            cuda_glwe_dot_product_with_clear_one_to_many(&d_input_glwe, &d_clear_matrix, d_output_lwe, &streams);
            cuda_lwe_ciphertext_add_assign_async(d_accum_lwe, d_output_lwe, streams);
        }
    }
    
    // TODO have a dot_reversed, as this allocates memory, depending on data size it might be slow
    // and inefficient
    pub fn dot(&self, other: &[Scalar]) -> EncryptedDotProductResult<Scalar> {
        assert_eq!(self.actual_len, other.len());
        let polynomial_size = self.data[0].polynomial_size();
        let glwe_size = self.data[0].glwe_size();
        let ciphertext_modulus = self.data[0].ciphertext_modulus();

        let mut dot_result =
            GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
        for (lhs, rhs) in self.data.iter().zip(other.chunks(polynomial_size.0)) {
            let tmp_rhs: Vec<_>;
            let rhs = if rhs.len() == polynomial_size.0 {
                rhs
            } else {
                tmp_rhs = rhs
                    .iter()
                    .copied()
                    .chain(core::iter::repeat(Scalar::ZERO).take(polynomial_size.0 - rhs.len()))
                    .collect();
                &tmp_rhs
            };
            crate::computations::add_dot_product_encrypted_clear(&mut dot_result, lhs, rhs);
        }

        EncryptedDotProductResult {
            data: crate::computations::extract_dot_product_as_lwe_ciphertext(&dot_result),
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SeededCompressedEncryptedVector<Scalar: UnsignedInteger> {
    pub data: Vec<crate::compression::CompressedModulusSwitchedSeededGlweCiphertext<Scalar>>,
    actual_len: usize,
}

impl<Scalar: UnsignedTorus> SeededCompressedEncryptedVector<Scalar> {
    pub fn new<KeyCont, NoiseDistribution, NoiseSeeder>(
        input: &[Scalar],
        glwe_secret_key: &GlweSecretKey<KeyCont>,
        bits_reserved_for_computation: usize,
        mod_switch_modulus: CiphertextModulusLog,
        noise_distribution: NoiseDistribution,
        ciphertext_modulus: CiphertextModulus<Scalar>,
        seeder: &mut NoiseSeeder,
    ) -> Self
    where
        Scalar: Encryptable<Uniform, NoiseDistribution>,
        KeyCont: Container<Element = Scalar>,
        NoiseDistribution: Distribution,
        NoiseSeeder: Seeder + ?Sized,
    {
        let actual_len = input.len();
        let polynomial_size = glwe_secret_key.polynomial_size().0;

        // Break input vector in chunks of polynomial size
        // If the last chunk does not fit into the polynomial size
        // it will be padded with zeros
        let glwes = input
            .chunks(polynomial_size)
            .map(|input| {
                let tmp_input: Vec<_>;
                let input = if input.len() == polynomial_size {
                    //The chunk has the same size as the poly size
                    input
                } else {
                    // Padding with zeros if needed
                    tmp_input = input
                        .iter()
                        .copied()
                        .chain(core::iter::repeat(Scalar::ZERO).take(polynomial_size - input.len()))
                        .collect();
                    &tmp_input
                };

                let seeded = crate::encryption::encrypt_slice_as_seeded_glwe(
                    input,
                    glwe_secret_key,
                    bits_reserved_for_computation,
                    noise_distribution,
                    ciphertext_modulus,
                    seeder,
                );

                crate::compression::CompressedModulusSwitchedSeededGlweCiphertext::compress(
                    &seeded,
                    mod_switch_modulus,
                )
            })
            .collect();

        Self {
            data: glwes,
            actual_len,
        }
    }

    pub fn decompress(&self) -> EncryptedVector<Scalar> {
        let data = self
            .data
            .iter()
            .map(|mod_switched| {
                let seeded_glwe = mod_switched.extract();
                seeded_glwe.decompress_into_glwe_ciphertext()
            })
            .collect();

        EncryptedVector {
            data,
            actual_len: self.actual_len,
        }
    }
}
