"""
Logarithmic number encoder for multiplicative operations.

Mathematical basis:
- For positive a, b: log(a * b) = log(a) + log(b)
- In the magnitude subspace (projection onto direction):
    proj_mag(emb_a) + proj_mag(emb_b) = proj_mag(emb_{a*b})

Sign is tracked separately in an orthogonal direction:
    sign(a * b) = sign(a) * sign(b)

The full embedding is magnitude + sign; the homomorphism is exact in real
arithmetic for the magnitude projection and approximate under IEEE-754.

Zero is handled explicitly by encoding it as the zero vector.

Reference: Flux Mathematics textbook, Chapter 8
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from typing import Tuple, Literal


class LogarithmicNumberEncoder(eqx.Module):
    """
    Encode numbers logarithmically for multiplication.

    The magnitude projection satisfies:
        log_mag(a * b) = log_mag(a) + log_mag(b)

    Sign is handled separately in an orthogonal direction:
        sign(a * b) = sign(a) * sign(b)
    Zero is encoded as the zero vector.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    log_scale : float
        Scale factor for log values.
        log_scale=20.0 handles numbers up to e^20 ~ 485 million.
    seed : int
        Random seed for direction vectors (only used if basis="random_orthonormal").
    basis : str
        "canonical" (default): Use e0 for magnitude, e1 for sign, decode via indexing.
            Error is minimal (log/exp precision only).
        "random_orthonormal": Use random orthonormal vectors, decode via dot product.
            Error accumulates over dim operations.
    """

    direction: jax.Array       # Unit direction for log magnitude
    sign_direction: jax.Array  # Orthogonal direction for sign
    log_scale: float
    dim: int
    basis: str
    epsilon: float = 1e-10     # Floor for log to handle zero

    def __init__(
        self,
        dim: int = 256,
        log_scale: float = 20.0,
        seed: int = 42,
        basis: Literal["canonical", "random_orthonormal"] = "canonical",
    ):
        self.log_scale = log_scale
        self.dim = dim
        self.basis = basis

        if basis == "canonical":
            # e0 for magnitude, e1 for sign
            direction = jnp.zeros(dim)
            direction = direction.at[0].set(1.0)
            self.direction = direction

            sign_direction = jnp.zeros(dim)
            sign_direction = sign_direction.at[1].set(1.0)
            self.sign_direction = sign_direction
        else:
            # Random orthonormal vectors (old behavior)
            key = jax.random.PRNGKey(seed)
            k1, k2 = jax.random.split(key)

            direction = jax.random.normal(k1, (dim,))
            self.direction = direction / jnp.linalg.norm(direction)

            # Gram-Schmidt for sign direction
            sign_dir = jax.random.normal(k2, (dim,))
            sign_dir = sign_dir - jnp.dot(sign_dir, self.direction) * self.direction
            self.sign_direction = sign_dir / jnp.linalg.norm(sign_dir)

    def encode_number(self, n: float) -> jax.Array:
        """
        Encode a number to a logarithmic embedding.

        Parameters
        ----------
        n : float
            The numeric value to encode.

        Returns
        -------
        jax.Array
            Logarithmic embedding, shape [dim].
        """
        sign = jnp.sign(n)
        abs_n = jnp.abs(n)

        is_zero = abs_n < self.epsilon
        safe_n = jnp.where(is_zero, 1.0, abs_n)

        log_value = jnp.log(safe_n)
        log_normalized = log_value / self.log_scale

        if self.basis == "canonical":
            # x[0] = log_normalized, x[1] = sign
            emb = jnp.zeros(self.dim)
            emb = emb.at[0].set(log_normalized)
            emb = emb.at[1].set(sign)
            return jnp.where(is_zero, jnp.zeros(self.dim), emb)
        else:
            magnitude_emb = log_normalized * self.direction
            sign_emb = sign * self.sign_direction * 0.5
            emb = magnitude_emb + sign_emb
            return jnp.where(is_zero, jnp.zeros_like(self.direction), emb)

    def encode_string(self, s: str) -> jax.Array:
        """
        Encode a string representation of a number.

        Parameters
        ----------
        s : str
            String representation (e.g., "42", "-123", "3.14").

        Returns
        -------
        jax.Array
            Logarithmic embedding, shape [dim].
        """
        try:
            value = float(s.strip())
        except (ValueError, AttributeError):
            value = 0.0
        return self.encode_number(value)

    def decode(self, emb: jax.Array) -> float:
        """
        Decode a logarithmic embedding back to a number.

        Parameters
        ----------
        emb : jax.Array
            Logarithmic embedding, shape [dim].

        Returns
        -------
        float
            The decoded number.
        """
        if self.basis == "canonical":
            # Direct indexing: x[0] = log_normalized, x[1] = sign
            emb_norm = jnp.linalg.norm(emb)
            is_zero = emb_norm < self.epsilon

            log_normalized = emb[0]
            log_value = log_normalized * self.log_scale
            magnitude = jnp.exp(log_value)

            # Sign is stored exactly as ±1 in x[1]
            sign = jnp.sign(emb[1])
            sign = jnp.where(sign == 0, 1.0, sign)

            value = sign * magnitude
            return float(jnp.where(is_zero, 0.0, value))
        else:
            emb_norm = jnp.linalg.norm(emb)
            is_zero = emb_norm < self.epsilon

            finite_emb = jnp.where(jnp.isfinite(emb), emb, 0.0)

            log_normalized = jnp.dot(finite_emb, self.direction)
            log_value = log_normalized * self.log_scale
            magnitude = jnp.exp(log_value)

            has_inf = jnp.any(jnp.isinf(emb))
            magnitude = jnp.where(has_inf, jnp.inf, magnitude)

            sign = self._extract_sign(finite_emb)

            value = sign * magnitude
            return float(jnp.where(is_zero, 0.0, value))

    def multiply(self, emb_a: jax.Array, emb_b: jax.Array) -> jax.Array:
        """
        Multiply two numbers in embedding space.

        This adds the magnitude components in log space:
        log(a * b) = log(a) + log(b)

        For signs: (+)(+)=+, (+)(-)=-, (-)(+)=-, (-)(-)=+

        Parameters
        ----------
        emb_a, emb_b : jax.Array
            Logarithmic embeddings of the operands.

        Returns
        -------
        jax.Array
            Logarithmic embedding of the product.
        """
        zero_a = self._is_zero_embedding(emb_a)
        zero_b = self._is_zero_embedding(emb_b)

        if self.basis == "canonical":
            # x[0] = log_mag, x[1] = sign
            result_log_mag = emb_a[0] + emb_b[0]  # log(a) + log(b) = log(a*b)

            sign_a = jnp.sign(emb_a[1])
            sign_b = jnp.sign(emb_b[1])
            sign_a = jnp.where(sign_a == 0, 1.0, sign_a)
            sign_b = jnp.where(sign_b == 0, 1.0, sign_b)
            result_sign = sign_a * sign_b

            result = jnp.zeros(self.dim)
            result = result.at[0].set(result_log_mag)
            result = result.at[1].set(result_sign)
        else:
            mag_a = jnp.dot(emb_a, self.direction) * self.direction
            mag_b = jnp.dot(emb_b, self.direction) * self.direction
            result_mag = mag_a + mag_b  # log(a) + log(b) = log(a*b)

            sign_a = self._extract_sign(emb_a)
            sign_b = self._extract_sign(emb_b)
            result_sign = sign_a * sign_b

            result = result_mag + result_sign * self.sign_direction * 0.5

        return jnp.where(zero_a | zero_b, jnp.zeros_like(result), result)

    def divide(self, emb_a: jax.Array, emb_b: jax.Array) -> jax.Array:
        """
        Divide two numbers in embedding space.

        This subtracts the magnitude components in log space:
        log(a / b) = log(a) - log(b)

        Parameters
        ----------
        emb_a, emb_b : jax.Array
            Logarithmic embeddings of numerator and denominator.

        Returns
        -------
        jax.Array
            Logarithmic embedding of the quotient.
        """
        zero_a = self._is_zero_embedding(emb_a)
        zero_b = self._is_zero_embedding(emb_b)

        if self.basis == "canonical":
            # x[0] = log_mag, x[1] = sign
            result_log_mag = emb_a[0] - emb_b[0]  # log(a) - log(b) = log(a/b)

            sign_a = jnp.sign(emb_a[1])
            sign_b = jnp.sign(emb_b[1])
            sign_a = jnp.where(sign_a == 0, 1.0, sign_a)
            sign_b = jnp.where(sign_b == 0, 1.0, sign_b)
            result_sign = sign_a * sign_b

            result = jnp.zeros(self.dim)
            result = result.at[0].set(result_log_mag)
            result = result.at[1].set(result_sign)

            inf_emb = self._inf_embedding(sign_a)
        else:
            mag_a = jnp.dot(emb_a, self.direction) * self.direction
            mag_b = jnp.dot(emb_b, self.direction) * self.direction
            result_mag = mag_a - mag_b  # log(a) - log(b) = log(a/b)

            sign_a = self._extract_sign(emb_a)
            sign_b = self._extract_sign(emb_b)
            result_sign = sign_a * sign_b

            result = result_mag + result_sign * self.sign_direction * 0.5
            inf_emb = self._inf_embedding(sign_a)

        return jnp.where(
            zero_b,
            inf_emb,
            jnp.where(zero_a, jnp.zeros_like(result), result),
        )

    def _is_zero_embedding(self, emb: jax.Array) -> jax.Array:
        return jnp.linalg.norm(emb) < self.epsilon

    def _extract_sign(self, emb: jax.Array) -> jax.Array:
        """Extract sign from embedding (for random_orthonormal basis)."""
        sign_proj = jnp.dot(emb, self.sign_direction) / 0.5
        sign = jnp.sign(sign_proj)
        sign = jnp.where(jnp.abs(sign_proj) < 0.1, 1.0, sign)
        sign = jnp.where(sign == 0, 1.0, sign)
        return sign

    def _inf_embedding(self, sign: jax.Array) -> jax.Array:
        """Create an embedding representing infinity with given sign."""
        if self.basis == "canonical":
            emb = jnp.zeros(self.dim)
            emb = emb.at[0].set(1e6)  # Large log value
            emb = emb.at[1].set(sign)
            return emb
        else:
            inf_log = jnp.array(1e6, dtype=self.direction.dtype)
            mag_emb = inf_log * self.direction
            sign_emb = sign * self.sign_direction * 0.5
            return mag_emb + sign_emb


def verify_multiplication_theorem(
    encoder: LogarithmicNumberEncoder,
    a: float,
    b: float,
    tol: float = 1e-4,
) -> bool:
    """
    Verify that log-magnitude projections add for multiplication.

    Parameters
    ----------
    encoder : LogarithmicNumberEncoder
        The encoder.
    a, b : float
        Numbers to test.
    tol : float
        Absolute tolerance on the log-magnitude projection.

    Returns
    -------
    bool
        True if the multiplication theorem holds.
    """
    if a == 0 or b == 0:
        return True

    emb_a = encoder.encode_number(a)
    emb_b = encoder.encode_number(b)
    emb_product = encoder.encode_number(a * b)

    computed = encoder.multiply(emb_a, emb_b)

    if encoder.basis == "canonical":
        computed_log_mag = computed[0]
        expected_log_mag = emb_product[0]
    else:
        computed_log_mag = jnp.dot(computed, encoder.direction)
        expected_log_mag = jnp.dot(emb_product, encoder.direction)

    abs_error = jnp.abs(computed_log_mag - expected_log_mag)
    return bool(abs_error < tol)


def verify_division_theorem(
    encoder: LogarithmicNumberEncoder,
    a: float,
    b: float,
    tol: float = 1e-4,
) -> bool:
    """
    Verify that log-magnitude projections subtract for division.

    Parameters
    ----------
    encoder : LogarithmicNumberEncoder
        The encoder.
    a, b : float
        Numbers to test (b must be non-zero).
    tol : float
        Absolute tolerance on the log-magnitude projection.

    Returns
    -------
    bool
        True if the division theorem holds.
    """
    if a == 0 or b == 0:
        return True

    emb_a = encoder.encode_number(a)
    emb_b = encoder.encode_number(b)
    emb_quotient = encoder.encode_number(a / b)

    computed = encoder.divide(emb_a, emb_b)

    if encoder.basis == "canonical":
        computed_log_mag = computed[0]
        expected_log_mag = emb_quotient[0]
    else:
        computed_log_mag = jnp.dot(computed, encoder.direction)
        expected_log_mag = jnp.dot(emb_quotient, encoder.direction)

    abs_error = jnp.abs(computed_log_mag - expected_log_mag)
    return bool(abs_error < tol)


if __name__ == "__main__":
    print("LogarithmicNumberEncoder demo")

    print("\n=== Canonical basis (default) ===")
    encoder = LogarithmicNumberEncoder(dim=256, log_scale=20.0, basis="canonical")

    print("\nRound-trip (encode -> decode):")
    test_numbers = [1, 2, 10, 42, 100, 1000, 10000, -42, -1000]
    for n in test_numbers:
        emb = encoder.encode_number(n)
        recovered = encoder.decode(emb)
        rel_error = abs(recovered - n) / abs(n) if n != 0 else 0
        print(f"  {n:>6} -> encode -> decode -> {recovered:>10.4f} (rel error: {rel_error:.2e})")

    print("\nMultiplication via embedding space:")
    mul_tests = [(6, 7), (12, 5), (100, 50), (123, 456), (-6, 7), (-6, -7)]
    for a, b in mul_tests:
        emb_a = encoder.encode_number(a)
        emb_b = encoder.encode_number(b)
        result_emb = encoder.multiply(emb_a, emb_b)
        result = encoder.decode(result_emb)
        expected = a * b
        rel_error = abs(result - expected) / abs(expected) if expected != 0 else 0
        status = "PASS" if rel_error < 1e-6 else "FAIL"
        print(f"  {a:>4} * {b:>4} = {result:>10.2f} (expected: {expected:>8}, rel error: {rel_error:.2e}) [{status}]")

    print("\nDivision via embedding space:")
    div_tests = [(42, 6), (100, 4), (1000, 8), (56088, 123), (-42, 6), (-42, -6)]
    for a, b in div_tests:
        emb_a = encoder.encode_number(a)
        emb_b = encoder.encode_number(b)
        result_emb = encoder.divide(emb_a, emb_b)
        result = encoder.decode(result_emb)
        expected = a / b
        rel_error = abs(result - expected) / abs(expected) if expected != 0 else 0
        status = "PASS" if rel_error < 1e-6 else "FAIL"
        print(f"  {a:>6} / {b:>4} = {result:>10.4f} (expected: {expected:>10.4f}, rel error: {rel_error:.2e}) [{status}]")

    print("\n=== Random orthonormal basis (old behavior) ===")
    encoder_old = LogarithmicNumberEncoder(dim=256, log_scale=20.0, basis="random_orthonormal", seed=42)

    print("\nMultiplication via embedding space:")
    for a, b in mul_tests[:4]:
        emb_a = encoder_old.encode_number(a)
        emb_b = encoder_old.encode_number(b)
        result_emb = encoder_old.multiply(emb_a, emb_b)
        result = encoder_old.decode(result_emb)
        expected = a * b
        rel_error = abs(result - expected) / abs(expected) if expected != 0 else 0
        status = "PASS" if rel_error < 0.01 else "FAIL"
        print(f"  {a:>4} * {b:>4} = {result:>10.2f} (expected: {expected:>8}, rel error: {rel_error:.2e}) [{status}]")
