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
    let ciphertext_modulus_bit_count = 31usize;
    // TODO: For now very small noise, find secure noise
    let glwe_encryption_noise_distribution = DynamicDistribution::new_gaussian_from_std_dev(
        StandardDev(2.0f64.powi(2) / 2.0f64.powi(ciphertext_modulus_bit_count as i32)),
    );

    let ciphertext_modulus =
        CiphertextModulus::try_new_power_of_2(ciphertext_modulus_bit_count).unwrap();
    let mod_switch_bit_count = ciphertext_modulus_bit_count - 1;
    let mod_switch_modulus = CiphertextModulusLog(mod_switch_bit_count);
    let bits_reserved_for_computation = 12;

    let data: Vec<Scalar> = (0..polynomial_size.0 * 2 + 17)
        .map(|x| x as Scalar % 2)
        .collect();

    // This could be a method to generate a private key object
    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    let mut secret_rng = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());

    let glwe_secret_key = allocate_and_generate_new_binary_glwe_secret_key(
        encryption_glwe_dimension,
        polynomial_size,
        &mut secret_rng,
    );

    let glwe_secret_key_as_lwe_secret_key = glwe_secret_key.as_lwe_secret_key();

    let compression_params = compression::CompressionKeyParameters {
        packing_ks_level: DecompositionLevelCount(2),
        packing_ks_base_log: DecompositionBaseLog(8),
        packing_ks_polynomial_size: PolynomialSize(256),
        packing_ks_glwe_dimension: GlweDimension(5),
        lwe_per_glwe: LweCiphertextCount(256),
        packing_ciphertext_modulus: ciphertext_modulus,
        storage_log_modulus: CiphertextModulusLog(mod_switch_bit_count),
        packing_ks_key_noise_distribution: DynamicDistribution::new_gaussian_from_std_dev(
            StandardDev(2.0f64.powi(2) / 2.0f64.powi(ciphertext_modulus_bit_count as i32)),
        ),
    };

    let (post_compression_glwe_secret_key, compression_key) =
        compression::CompressionKey::new(&glwe_secret_key_as_lwe_secret_key, compression_params);

    // This would be a method to encrypt a numpy array using a private key object
    let seeded_encrypted_vector = ml::SeededCompressedEncryptedVector::new(
        &data,
        &glwe_secret_key,
        bits_reserved_for_computation,
        mod_switch_modulus,
        glwe_encryption_noise_distribution,
        ciphertext_modulus,
        seeder,
    );

    // A method to serialize a ciphertext (and compress it?)
    let serialized = bincode::serialize(&seeded_encrypted_vector).unwrap();

    // A method to deserialize a ciphertext
    let deserialized: ml::SeededCompressedEncryptedVector<Scalar> =
        bincode::deserialize(&serialized).unwrap();

    let encrypted_vector = deserialized.decompress();

    // A method to decrypt a ciphertext
    let decrypted = encrypted_vector.decrypt(&glwe_secret_key, bits_reserved_for_computation);

    assert_eq!(&decrypted, &data);

    let clear_2: Vec<Scalar> = (0..data.len()).map(|x| x as Scalar % 3).collect();

    let result = encrypted_vector.dot(&clear_2);

    let compressed_results = compression_key.compress_ciphertexts_into_list(&[result]);

    let mut clear_dot = 0;

    for (lhs, rhs) in data.iter().copied().zip(clear_2.iter().copied()) {
        clear_dot += lhs * rhs;
    }

    let extracted: Vec<_> = compressed_results
        .into_iter()
        .map(|compressed| compressed.extract())
        .collect();

    // We have a single result here but we could have more
    assert_eq!(extracted.len(), 1);

    let result = extracted.into_iter().next().unwrap();

    let decrypted_dot = encryption::decrypt_glwe(
        &post_compression_glwe_secret_key,
        &result,
        bits_reserved_for_computation,
    );

    let decrypted_dot = decrypted_dot[0];
    // With all the noise in this implem we take a small margin

    let diff = decrypted_dot.abs_diff(clear_dot);

    const MAX_DIFF: Scalar = 10;
    assert!(
        diff < MAX_DIFF,
        "Result has a deviation bigger than {MAX_DIFF}, expected {clear_dot}, got {decrypted_dot}",
    );

    assert_eq!(
        decrypted_dot, clear_dot,
        "Result is not exact and has some computation noise"
    );
}
