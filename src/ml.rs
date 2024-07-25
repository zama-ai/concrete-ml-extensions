use tfhe::core_crypto::commons::math::random::{Distribution, Uniform};
use tfhe::core_crypto::prelude::*;

pub struct EncryptedVector<Scalar: UnsignedInteger> {
    // TODO: manage the data via a GlweCiphertextList
    data: Vec<GlweCiphertextOwned<Scalar>>,
}

pub struct SeededCompressedEncryptedVector<Scalar: UnsignedInteger> {
    // TODO: manage the data via a GlweCiphertextList
    data: Vec<crate::compression::CompressedModulusSwitchedSeededGlweCiphertext<Scalar>>,
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

        Self { data: glwes }
    }
}
