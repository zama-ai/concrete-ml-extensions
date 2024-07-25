#![allow(clippy::excessive_precision)]

use tfhe::core_crypto::prelude::*;

mod compression;
mod computations;
mod encryption;

fn main() {
    let encryption_glwe_dimension = GlweDimension(1);
    let polynomial_size = PolynomialSize(2048);
    // TODO: For now very small noise, find secure noise
    let glwe_encryption_noise_distribution =
        DynamicDistribution::new_gaussian_from_std_dev(StandardDev(3.725290298461914e-09));

    let ciphertext_modulus_bit_count = 31;
    let ciphertext_modulus =
        CiphertextModulus::try_new_power_of_2(ciphertext_modulus_bit_count).unwrap();
    let mod_switch_bit_count = ciphertext_modulus_bit_count - 1;
    let mod_switch_modulus = CiphertextModulusLog(mod_switch_bit_count);
    let bits_reserved_for_computation = 15;

    let data: Vec<u32> = (0..polynomial_size.0).map(|x| x as u32 % 2).collect();

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

    let glwe_secret_key_as_lwe_secret_key = glwe_secret_key.as_lwe_secret_key();

    // Other sanity check without seeding
    {
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

    let seeded_glwe = encryption::encrypt_slice_as_seeded_glwe(
        &data,
        &glwe_secret_key,
        bits_reserved_for_computation,
        glwe_encryption_noise_distribution,
        ciphertext_modulus,
        seeder,
    );

    let mod_switched = compression::CompressedModulusSwitchedSeededGlweCiphertext::compress(
        &seeded_glwe,
        mod_switch_modulus,
    );

    let serialized = bincode::serialize(&mod_switched).unwrap();

    let deserialized = bincode::deserialize(&serialized);

    let demod_switched = deserialized.extract();

    let glwe = demod_switched.decompress_into_glwe_ciphertext();

    let decrypted =
        encryption::decrypt_glwe(&glwe_secret_key, &glwe, bits_reserved_for_computation);

    assert_eq!(&decrypted, &data);

    let clear_2: Vec<u32> = (0..polynomial_size.0).map(|x| x as u32 % 3).collect();

    let mut result = glwe.clone();

    computations::dot_product_encrypted_clear(&mut result, &glwe, &clear_2);

    let mut clear_dot = 0;

    for (lhs, rhs) in data.iter().copied().zip(clear_2.iter().copied()) {
        clear_dot += lhs * rhs;
    }

    let result_as_lwe = computations::extract_dot_product_as_lwe_ciphertext(&result);

    let decrypted_dot = encryption::decrypt_lwe(
        &glwe_secret_key_as_lwe_secret_key,
        &result_as_lwe,
        bits_reserved_for_computation,
    );

    assert_eq!(decrypted_dot, clear_dot);
}
