use tfhe::core_crypto::commons::math::random::CompressionSeed;
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
        let uncompressed_ciphertext_modulus = ct.ciphertext_modulus();

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

        let seeded_glwe = SeededGlweCiphertextOwned::from_container(
            container,
            self.glwe_dimension().to_glwe_size(),
            self.compression_seed,
            self.uncompressed_ciphertext_modulus(),
        );

        seeded_glwe
    }
}
