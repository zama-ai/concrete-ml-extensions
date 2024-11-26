use tfhe::core_crypto::prelude::*;
//use tfhe::core_crypto::gpu::bin
//cuda_fourier_polynomial_mul

#[allow(dead_code)]
/// Computes a dot product in a GLWE's last coefficient, with the clear input in the normally
/// indexed order
pub fn dot_product_encrypted_clear<Scalar, InputCont, OutputCont>(
    out: &mut GlweCiphertext<OutputCont>,
    enc: &GlweCiphertext<InputCont>,
    clear: &[Scalar],
) where
    Scalar: UnsignedInteger,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    let clear = {
        let mut tmp = clear.to_vec();
        tmp.reverse();
        tmp
    };

    dot_product_encrypted_inverted_clear(out, enc, &clear)
}

#[allow(dead_code)]
/// Computes a dot product in a GLWE's last coefficient, with the clear input in revese indexed
/// order
pub fn dot_product_encrypted_inverted_clear<Scalar, InputCont, OutputCont>(
    out: &mut GlweCiphertext<OutputCont>,
    enc: &GlweCiphertext<InputCont>,
    clear: &[Scalar],
) where
    Scalar: UnsignedInteger,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    use polynomial_algorithms;
    assert_eq!(out.glwe_size(), enc.glwe_size());
    assert_eq!(out.polynomial_size(), enc.polynomial_size());
    assert_eq!(clear.len(), enc.polynomial_size().0);

    let clear_as_polynomial = Polynomial::from_container(clear);

    for (mut out_poly, in_poly) in out
        .as_mut_polynomial_list()
        .iter_mut()
        .zip(enc.as_polynomial_list().iter())
    {
        polynomial_algorithms::polynomial_wrapping_mul(
            &mut out_poly,
            &in_poly,
            &clear_as_polynomial,
        );
    }
}

/// Computes a dot product in a GLWE's last coefficient, with the clear input in the normally
/// indexed order
pub fn add_dot_product_encrypted_clear<Scalar, InputCont, OutputCont>(
    out: &mut GlweCiphertext<OutputCont>,
    enc: &GlweCiphertext<InputCont>,
    clear: &[Scalar],
) where
    Scalar: UnsignedInteger,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    let clear = {
        let mut tmp = clear.to_vec();
        tmp.reverse();
        tmp
    };

    add_dot_product_encrypted_inverted_clear(out, enc, &clear)
}

/// Computes a dot product in a GLWE's last coefficient, with the clear input in revese indexed
/// order
pub fn add_dot_product_encrypted_inverted_clear<Scalar, InputCont, OutputCont>(
    out: &mut GlweCiphertext<OutputCont>,
    enc: &GlweCiphertext<InputCont>,
    clear: &[Scalar],
) where
    Scalar: UnsignedInteger,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    use polynomial_algorithms;
    assert_eq!(out.glwe_size(), enc.glwe_size());
    assert_eq!(out.polynomial_size(), enc.polynomial_size());
    assert_eq!(clear.len(), enc.polynomial_size().0);

    let clear_as_polynomial = Polynomial::from_container(clear);

    for (mut out_poly, in_poly) in out
        .as_mut_polynomial_list()
        .iter_mut()
        .zip(enc.as_polynomial_list().iter())
    {
        polynomial_algorithms::polynomial_wrapping_add_mul_assign(
            &mut out_poly,
            &in_poly,
            &clear_as_polynomial,
        );
    }
}

pub fn extract_dot_product_as_lwe_ciphertext<Scalar, InputCont>(
    glwe: &GlweCiphertext<InputCont>,
) -> LweCiphertextOwned<Scalar>
where
    Scalar: UnsignedInteger,
    InputCont: Container<Element = Scalar>,
{
    let mut lwe_ciphertext = LweCiphertext::new(
        Scalar::ZERO,
        glwe.glwe_size()
            .to_glwe_dimension()
            .to_equivalent_lwe_dimension(glwe.polynomial_size())
            .to_lwe_size(),
        glwe.ciphertext_modulus(),
    );

    extract_lwe_sample_from_glwe_ciphertext(
        glwe,
        &mut lwe_ciphertext,
        MonomialDegree(glwe.polynomial_size().0 - 1),
    );

    lwe_ciphertext
}
