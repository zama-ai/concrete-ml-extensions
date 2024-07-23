use tfhe::core_crypto::prelude::*;

mod encryption;

fn main() {
    let encryption_glwe_dimension = GlweDimension(1);
    let polynomial_size = PolynomialSize(2048);
    // TODO: For now no noise
    let glwe_encryption_noise_distribution =
        DynamicDistribution::new_gaussian_from_std_dev(StandardDev(0.0));

    let ciphertext_modulus = CiphertextModulus::try_new_power_of_2(22).unwrap();
    let bits_reserved_for_computation = 10;
    let modulus_for_computation = 1u32 << bits_reserved_for_computation;

    let data: Vec<u32> = (0..polynomial_size.0)
        .map(|x| x as u32 % modulus_for_computation)
        .collect();

    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    let mut secret_rng = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
    let mut encryption_rng =
        EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

    let glwe_secret_key = allocate_and_generate_new_binary_glwe_secret_key(
        encryption_glwe_dimension,
        polynomial_size,
        &mut secret_rng,
    );

    let glwe = encryption::encrypt_slice_as_glwe(
        &data,
        &glwe_secret_key,
        bits_reserved_for_computation,
        glwe_encryption_noise_distribution,
        ciphertext_modulus,
        &mut encryption_rng,
    );

    let decrypted =
        encryption::decrypt_glwe(&glwe_secret_key, &glwe, bits_reserved_for_computation);

    assert_eq!(&decrypted, &data);
}
