use tfhe::core_crypto::commons::ciphertext_modulus::CiphertextModulusKind;
use tfhe::core_crypto::commons::math::random::{Distribution, Uniform};
use tfhe::core_crypto::prelude::*;

pub fn encryption_delta<Scalar: UnsignedInteger>(
    bits_reserved_for_computation: usize,
    ciphertext_modulus: CiphertextModulus<Scalar>,
) -> Scalar {
    let modulus_for_computations = Scalar::ONE << bits_reserved_for_computation;
    match ciphertext_modulus.kind() {
        CiphertextModulusKind::Native => {
            ((Scalar::ONE << (Scalar::BITS - 1)) / modulus_for_computations) << 1
        }
        CiphertextModulusKind::NonNativePowerOfTwo => {
            let custom_mod = ciphertext_modulus
                .get_custom_modulus_as_optional_scalar()
                .unwrap();
            custom_mod / modulus_for_computations
        }
        CiphertextModulusKind::Other => todo!("Only power of 2 moduli are supported"),
    }
}

pub fn encode_data_for_encryption<Scalar, OutputCont>(
    input: &[Scalar],
    plaintext_list: &mut PlaintextList<OutputCont>,
    bits_reserved_for_computation: usize,
    ciphertext_modulus: CiphertextModulus<Scalar>,
) where
    Scalar: UnsignedInteger,
    OutputCont: ContainerMut<Element = Scalar>,
{
    assert!(input.len() <= plaintext_list.entity_count());

    let delta = encryption_delta(bits_reserved_for_computation, ciphertext_modulus);

    for (plain, input) in plaintext_list.iter_mut().zip(input.iter()) {
        *plain.0 = (*input) * delta;
    }
}

#[allow(dead_code)]
pub fn encrypt_slice_as_glwe<Scalar, KeyCont, NoiseDistribution>(
    input: &[Scalar],
    glwe_secret_key: &GlweSecretKey<KeyCont>,
    bits_reserved_for_computation: usize,
    noise_distribution: NoiseDistribution,
    ciphertext_modulus: CiphertextModulus<Scalar>,
    generator: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>,
) -> GlweCiphertextOwned<Scalar>
where
    NoiseDistribution: Distribution,
    Scalar: UnsignedInteger + Encryptable<Uniform, NoiseDistribution>,
    KeyCont: Container<Element = Scalar>,
{
    assert!(input.len() <= glwe_secret_key.polynomial_size().0);

    let mut plaintext_list = PlaintextList::new(
        Scalar::ZERO,
        PlaintextCount(glwe_secret_key.polynomial_size().0),
    );

    encode_data_for_encryption(
        input,
        &mut plaintext_list,
        bits_reserved_for_computation,
        ciphertext_modulus,
    );

    let mut glwe = GlweCiphertext::new(
        Scalar::ZERO,
        glwe_secret_key.glwe_dimension().to_glwe_size(),
        glwe_secret_key.polynomial_size(),
        ciphertext_modulus,
    );

    encrypt_glwe_ciphertext(
        glwe_secret_key,
        &mut glwe,
        &plaintext_list,
        noise_distribution,
        generator,
    );

    glwe
}

pub fn encrypt_slice_as_seeded_glwe<Scalar, KeyCont, NoiseDistribution, NoiseSeeder>(
    input: &[Scalar],
    glwe_secret_key: &GlweSecretKey<KeyCont>,
    bits_reserved_for_computation: usize,
    noise_distribution: NoiseDistribution,
    ciphertext_modulus: CiphertextModulus<Scalar>,
    seeder: &mut NoiseSeeder,
) -> SeededGlweCiphertextOwned<Scalar>
where
    NoiseDistribution: Distribution,
    Scalar: UnsignedInteger + Encryptable<Uniform, NoiseDistribution>,
    KeyCont: Container<Element = Scalar>,
    NoiseSeeder: Seeder + ?Sized,
{
    assert!(input.len() <= glwe_secret_key.polynomial_size().0);

    let mut plaintext_list = PlaintextList::new(
        Scalar::ZERO,
        PlaintextCount(glwe_secret_key.polynomial_size().0),
    );

    encode_data_for_encryption(
        input,
        &mut plaintext_list,
        bits_reserved_for_computation,
        ciphertext_modulus,
    );

    let mut seeded_glwe = SeededGlweCiphertext::new(
        Scalar::ZERO,
        glwe_secret_key.glwe_dimension().to_glwe_size(),
        glwe_secret_key.polynomial_size(),
        seeder.seed().into(),
        ciphertext_modulus,
    );

    encrypt_seeded_glwe_ciphertext(
        glwe_secret_key,
        &mut seeded_glwe,
        &plaintext_list,
        noise_distribution,
        seeder,
    );

    seeded_glwe
}

pub fn decrypt_glwe<Scalar, InputCont, KeyCont>(
    glwe_secret_key: &GlweSecretKey<KeyCont>,
    glwe: &GlweCiphertext<InputCont>,
    bits_reserved_for_computation: usize,
) -> Vec<Scalar>
where
    Scalar: UnsignedTorus,
    InputCont: Container<Element = Scalar>,
    KeyCont: Container<Element = Scalar>,
{
    let mut decrypted = PlaintextList::new(Scalar::ZERO, PlaintextCount(glwe.polynomial_size().0));
    let mut decoded = vec![Scalar::ZERO; decrypted.plaintext_count().0];
    decrypt_glwe_ciphertext(glwe_secret_key, glwe, &mut decrypted);

    let ciphertext_modulus = glwe.ciphertext_modulus();
    let delta = encryption_delta(bits_reserved_for_computation, ciphertext_modulus);

    let decomposer = SignedDecomposer::new(
        DecompositionBaseLog(
            bits_reserved_for_computation
                + ciphertext_modulus
                    .get_power_of_two_scaling_to_native_torus()
                    .ilog2() as usize,
        ),
        DecompositionLevelCount(1),
    );

    // Why the modulo ?
    for (decoded_value, decrypted_value) in decoded.iter_mut().zip(decrypted.iter()) {
        *decoded_value = (decomposer.closest_representable(*decrypted_value.0) / delta)
            % (Scalar::ONE << bits_reserved_for_computation);
    }

    decoded
}

pub fn decrypt_lwe<Scalar, InputCont, KeyCont>(
    lwe_secret_key: &LweSecretKey<KeyCont>,
    lwe: &LweCiphertext<InputCont>,
    bits_reserved_for_computation: usize,
) -> Scalar
where
    Scalar: UnsignedTorus,
    InputCont: Container<Element = Scalar>,
    KeyCont: Container<Element = Scalar>,
{
    let decrypted = decrypt_lwe_ciphertext(lwe_secret_key, lwe);

    let ciphertext_modulus = lwe.ciphertext_modulus();
    let delta = encryption_delta(bits_reserved_for_computation, ciphertext_modulus);

    let decomposer = SignedDecomposer::new(
        DecompositionBaseLog(
            bits_reserved_for_computation
                + ciphertext_modulus
                    .get_power_of_two_scaling_to_native_torus()
                    .ilog2() as usize,
        ),
        DecompositionLevelCount(1),
    );

    (decomposer.closest_representable(decrypted.0) / delta)
        % (Scalar::ONE << bits_reserved_for_computation)
}
