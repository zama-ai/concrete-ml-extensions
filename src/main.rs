#![allow(clippy::excessive_precision)]

use tfhe::core_crypto::prelude::*;

mod compression;
mod computations;
mod encryption;
mod ml;

fn main() {
    type Scalar = u32;
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

    let data: Vec<Scalar> = (0..polynomial_size.0 * 2 + 17)
        .map(|x| x as Scalar % 2)
        .collect();

    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    let mut secret_rng = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());

    let glwe_secret_key = allocate_and_generate_new_binary_glwe_secret_key(
        encryption_glwe_dimension,
        polynomial_size,
        &mut secret_rng,
    );

    let glwe_secret_key_as_lwe_secret_key = glwe_secret_key.as_lwe_secret_key();

    let seeded_encrypted_vector = ml::SeededCompressedEncryptedVector::new(
        &data,
        &glwe_secret_key,
        bits_reserved_for_computation,
        mod_switch_modulus,
        glwe_encryption_noise_distribution,
        ciphertext_modulus,
        seeder,
    );

    let serialized = bincode::serialize(&seeded_encrypted_vector).unwrap();

    let deserialized: ml::SeededCompressedEncryptedVector<Scalar> =
        bincode::deserialize(&serialized).unwrap();

    let encrypted_vector = deserialized.decompress();

    let decrypted = encrypted_vector.decrypt(&glwe_secret_key, bits_reserved_for_computation);

    assert_eq!(&decrypted, &data);

    let clear_2: Vec<Scalar> = (0..data.len()).map(|x| x as Scalar % 3).collect();

    let result = encrypted_vector.dot(&clear_2);

    let mut clear_dot = 0;

    for (lhs, rhs) in data.iter().copied().zip(clear_2.iter().copied()) {
        clear_dot += lhs * rhs;
    }

    let decrypted_dot = result.decrypt(
        &glwe_secret_key_as_lwe_secret_key,
        bits_reserved_for_computation,
    );

    assert_eq!(decrypted_dot, clear_dot);
}
