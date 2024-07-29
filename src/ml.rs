use tfhe::core_crypto::commons::math::random::{Distribution, Uniform};
use tfhe::core_crypto::prelude::*;

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
    // TODO: manage the data via a GlweCiphertextList
    data: Vec<GlweCiphertextOwned<Scalar>>,
    actual_len: usize,
}

// TODO: isn't there a way to do the mat-mult by overloading the scalar product
pub struct MatrixShape {
    x: usize,
    y: usize,
}

pub struct ClearMatrix<Scalar: UnsignedInteger> {
    // TODO: manage the data via a GlweCiphertextList
    data: Vec<Vec<Scalar>>,
    shape: MatrixShape,
}

pub struct EncryptedMatrix<Scalar: UnsignedInteger> {
    // TODO: manage the data via a GlweCiphertextList
    data: Vec<Vec<GlweCiphertextOwned<Scalar>>>,
    shape: MatrixShape,
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
    // TODO: manage the data via a GlweCiphertextList
    data: Vec<crate::compression::CompressedModulusSwitchedSeededGlweCiphertext<Scalar>>,
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
        let glwes = input
            .chunks(polynomial_size)
            .map(|input| {
                let tmp_input: Vec<_>;
                let input = if input.len() == polynomial_size {
                    input
                } else {
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
