use rayon::prelude::*;
use tfhe::core_crypto::commons::math::random::CompressionSeed;
use tfhe::core_crypto::entities::compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext;
use tfhe::core_crypto::entities::packed_integers::PackedIntegers;
use tfhe::core_crypto::fft_impl::common::modulus_switch;
use tfhe::core_crypto::prelude::*;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressedModulusSwitchedSeededGlweCiphertext<Scalar: UnsignedInteger> {
    packed_integers: PackedIntegers<Scalar>,
    // TODO: remove once the field is accessible in the PackedIntegers primitive of TFHE-rs
    log_modulus: CiphertextModulusLog,
    glwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    compression_seed: CompressionSeed,
    uncompressed_ciphertext_modulus: CiphertextModulus<Scalar>,
}

impl<Scalar: UnsignedTorus> CompressedModulusSwitchedSeededGlweCiphertext<Scalar> {
    pub fn glwe_dimension(&self) -> GlweDimension {
        self.glwe_dimension
    }
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.polynomial_size
    }
    pub fn uncompressed_ciphertext_modulus(&self) -> CiphertextModulus<Scalar> {
        self.uncompressed_ciphertext_modulus
    }

    /// Compresses a ciphertext by reducing its modulus
    /// This operation adds a lot of noise
    pub fn compress<Cont: Container<Element = Scalar>>(
        ct: &SeededGlweCiphertext<Cont>,
        log_modulus: CiphertextModulusLog,
    ) -> Self {
        let uncompressed_ciphertext_modulus: CiphertextModulus<Scalar> = ct.ciphertext_modulus();

        assert!(
            ct.ciphertext_modulus().is_power_of_two(),
            "Modulus switch compression doe not support non power of 2 input moduli",
        );

        let uncompressed_ciphertext_modulus_log =
            if uncompressed_ciphertext_modulus.is_native_modulus() {
                Scalar::BITS
            } else {
                uncompressed_ciphertext_modulus.get_custom_modulus().ilog2() as usize
            };

        let glwe_dimension = ct.glwe_size().to_glwe_dimension();
        let polynomial_size = ct.polynomial_size();

        assert!(
            log_modulus.0 <= uncompressed_ciphertext_modulus_log,
            "The log_modulus (={}) for modulus switch compression must be smaller than the uncompressed ciphertext_modulus_log (={})",
            log_modulus.0,
            uncompressed_ciphertext_modulus_log,
        );

        let compression_seed = ct.compression_seed();

        let modulus_switched: Vec<_> = ct
            .as_ref()
            .iter()
            .map(|a| modulus_switch(*a, log_modulus))
            .collect();

        let packed_integers = PackedIntegers::pack(&modulus_switched, log_modulus);

        Self {
            packed_integers,
            log_modulus,
            glwe_dimension,
            polynomial_size,
            compression_seed,
            uncompressed_ciphertext_modulus,
        }
    }

    /// Converts back a compressed ciphertext to its initial modulus
    /// The noise added during the compression stays in the output
    /// The output must got through a PBS to reduce the noise
    pub fn extract(&self) -> SeededGlweCiphertextOwned<Scalar> {
        let log_modulus = self.log_modulus.0;

        let container: Vec<_> = self
            .packed_integers
            .unpack()
            // Scaling
            .map(|a| a << (Scalar::BITS - log_modulus))
            .collect();

        assert_eq!(container.len(), self.polynomial_size().0);

        SeededGlweCiphertextOwned::from_container(
            container,
            self.glwe_dimension().to_glwe_size(),
            self.compression_seed,
            self.uncompressed_ciphertext_modulus(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompressionKeyParameters<Scalar: UnsignedInteger> {
    pub packing_ks_level: DecompositionLevelCount,
    pub packing_ks_base_log: DecompositionBaseLog,
    pub packing_ks_polynomial_size: PolynomialSize,
    pub packing_ks_glwe_dimension: GlweDimension,
    pub packing_ciphertext_modulus: CiphertextModulus<Scalar>,
    pub lwe_per_glwe: LweCiphertextCount,
    pub storage_log_modulus: CiphertextModulusLog,
    pub packing_ks_key_noise_distribution: DynamicDistribution<Scalar>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompressionKey<Scalar: UnsignedInteger> {
    pub packing_key_switching_key: LwePackingKeyswitchKeyOwned<Scalar>,
    pub lwe_per_glwe: LweCiphertextCount,
    pub storage_log_modulus: CiphertextModulusLog,
}

impl<Scalar: UnsignedTorus + Sync + Send> CompressionKey<Scalar> {
    pub fn new<InputKeyCont>(
        input_lwe_secret_key: &LweSecretKey<InputKeyCont>,
        params: CompressionKeyParameters<Scalar>,
    ) -> (GlweSecretKeyOwned<Scalar>, Self)
    where
        InputKeyCont: Container<Element = Scalar>,
    {
        let mut seeder = new_seeder();
        let seeder = seeder.as_mut();

        let mut secret_rng = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
        let mut encryption_rng =
            EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

        let post_packing_secret_key = allocate_and_generate_new_binary_glwe_secret_key(
            params.packing_ks_glwe_dimension,
            params.packing_ks_polynomial_size,
            &mut secret_rng,
        );

        let packing_key_switching_key = allocate_and_generate_new_lwe_packing_keyswitch_key(
            input_lwe_secret_key,
            &post_packing_secret_key,
            params.packing_ks_base_log,
            params.packing_ks_level,
            params.packing_ks_key_noise_distribution,
            params.packing_ciphertext_modulus,
            &mut encryption_rng,
        );

        let glwe_compression_key = CompressionKey {
            packing_key_switching_key,
            lwe_per_glwe: params.lwe_per_glwe,
            storage_log_modulus: params.storage_log_modulus,
        };

        (post_packing_secret_key, glwe_compression_key)
    }

    pub fn compress_ciphertexts_into_list(
        &self,
        ciphertexts: &[crate::ml::EncryptedDotProductResult<Scalar>],
    ) -> Vec<CompressedModulusSwitchedGlweCiphertext<Scalar>> {
        let lwe_pksk = &self.packing_key_switching_key;

        let polynomial_size = lwe_pksk.output_polynomial_size();
        let ciphertext_modulus: CiphertextModulus<Scalar> = lwe_pksk.ciphertext_modulus();
        let glwe_size = lwe_pksk.output_glwe_size();
        let lwe_size = lwe_pksk.input_key_lwe_dimension().to_lwe_size();

        let lwe_per_glwe = self.lwe_per_glwe;

        assert!(
            lwe_per_glwe.0 <= polynomial_size.0,
            "Cannot pack more than polynomial_size(={}) elements per glwe, {} requested",
            polynomial_size.0,
            lwe_per_glwe.0,
        );

        ciphertexts
            .par_chunks(lwe_per_glwe.0)
            .map(|ct_list| {
                let mut list: Vec<_> = vec![];

                for ct in ct_list {
                    let ct = ct.as_lwe();
                    assert_eq!(
                        lwe_size,
                        ct.lwe_size(),
                        "All ciphertexts do not have the same lwe size as the packing keyswitch key"
                    );

                    list.extend(ct.as_ref());
                }

                let list = LweCiphertextList::from_container(list, lwe_size, ciphertext_modulus);

                let bodies_count = LweCiphertextCount(ct_list.len());

                let mut out = GlweCiphertext::new(
                    Scalar::ZERO,
                    glwe_size,
                    polynomial_size,
                    ciphertext_modulus,
                );

                // TODO: add primitives to avoid having to use list primitives when possible
                par_keyswitch_lwe_ciphertext_list_and_pack_in_glwe_ciphertext(
                    lwe_pksk, &list, &mut out,
                );

                CompressedModulusSwitchedGlweCiphertext::compress(
                    &out,
                    self.storage_log_modulus,
                    bodies_count,
                )
            })
            .collect()
    }
}
