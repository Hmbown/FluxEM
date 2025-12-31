"""
Linear number encoder for additive operations.

Encodes numeric values into linear embeddings:
    encode(n) = n * (unit_direction / scale)

Linearity property:
    encode(a) + encode(b) = encode(a + b)
    encode(a) - encode(b) = encode(a - b)

This identity is exact in real arithmetic and approximate under IEEE-754
floating point.

Reference: Flux Mathematics textbook, Chapter 8
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from typing import Tuple, Optional, Literal


class NumberEncoder(eqx.Module):
    """
    Encode digit strings to LINEAR number embeddings.

    Key property: The output is linear in the numeric value:
    encode(n) = n * (direction / scale).
    This is a direct computation rather than a learned embedding.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    scale : float
        Scale factor to normalize embeddings.
        For numbers up to 100,000, scale=100000 keeps ||embed|| <= 1.
    seed : int
        Random seed for direction vector (only used if basis="random_orthonormal").
    basis : str
        "canonical" (default): Use e0 as direction, decode via indexing.
            Error is minimal (single float op).
        "random_orthonormal": Use random unit vector, decode via dot product.
            Error accumulates over dim operations.
    """

    direction: jax.Array  # Unit direction vector [dim]
    scale: float
    dim: int
    basis: str

    def __init__(
        self,
        dim: int = 256,
        scale: float = 100000.0,
        seed: int = 42,
        basis: Literal["canonical", "random_orthonormal"] = "canonical",
    ):
        self.scale = scale
        self.dim = dim
        self.basis = basis

        if basis == "canonical":
            # e0: first coordinate only
            direction = jnp.zeros(dim)
            direction = direction.at[0].set(1.0)
            self.direction = direction
        else:
            # Random unit vector (old behavior)
            key = jax.random.PRNGKey(seed)
            direction = jax.random.normal(key, (dim,))
            self.direction = direction / jnp.linalg.norm(direction)

    def encode_number(self, n: float) -> jax.Array:
        """
        Encode a number to a linear embedding.

        Parameters
        ----------
        n : float
            The numeric value.

        Returns
        -------
        jax.Array
            Linear embedding, shape [dim].
        """
        if self.basis == "canonical":
            # Direct indexing: x[0] = n/scale, rest zeros
            emb = jnp.zeros(self.dim)
            return emb.at[0].set(n / self.scale)
        else:
            return (n / self.scale) * self.direction

    def encode_string(self, digit_str: str) -> jax.Array:
        """
        Encode a digit string to a linear embedding.

        Parameters
        ----------
        digit_str : str
            String representation of a number (e.g., "42", "-123", "3.14").

        Returns
        -------
        jax.Array
            Linear embedding, shape [dim].
        """
        try:
            value = float(digit_str.strip())
        except (ValueError, AttributeError):
            value = 0.0
        return self.encode_number(value)

    def encode_bytes(self, byte_seq: jax.Array) -> jax.Array:
        """
        Encode a byte sequence representing a number.

        Parameters
        ----------
        byte_seq : jax.Array
            Byte sequence (ASCII values), shape [max_len].
            Padded with zeros.

        Returns
        -------
        jax.Array
            Linear embedding, shape [dim].
        """
        value = self._parse_number_from_bytes(byte_seq)
        if self.basis == "canonical":
            emb = jnp.zeros(self.dim)
            return emb.at[0].set(value / self.scale)
        else:
            return (value / self.scale) * self.direction

    def _parse_number_from_bytes(self, byte_seq: jax.Array) -> jax.Array:
        """
        Parse a number from a byte sequence using JAX operations.

        This implements place-value parsing:
        "123" = 1*100 + 2*10 + 3*1

        Handles negative numbers and ignores non-digit characters.
        """
        ZERO = ord('0')
        NINE = ord('9')
        MINUS = ord('-')

        is_digit = (byte_seq >= ZERO) & (byte_seq <= NINE)
        is_minus = byte_seq == MINUS

        first_nonzero_idx = jnp.argmax(byte_seq > 0)
        is_negative = byte_seq[first_nonzero_idx] == MINUS

        digit_values = jnp.where(is_digit, byte_seq - ZERO, 0)

        valid_positions = jnp.where(is_digit, jnp.arange(len(byte_seq)), len(byte_seq))
        first_digit_pos = jnp.min(valid_positions)
        valid_positions_for_max = jnp.where(is_digit, jnp.arange(len(byte_seq)), -1)
        last_digit_pos = jnp.max(valid_positions_for_max)

        n_digits = jnp.sum(is_digit)

        position_in_number = jnp.cumsum(is_digit) - 1
        place_values = jnp.where(
            is_digit,
            10.0 ** (n_digits - 1 - position_in_number),
            0.0
        )

        value = jnp.sum(digit_values * place_values)
        value = jnp.where(is_negative, -value, value)

        return value

    def decode(self, embedding: jax.Array) -> float:
        """
        Recover the number from a linear embedding.

        Parameters
        ----------
        embedding : jax.Array
            Linear embedding, shape [dim].

        Returns
        -------
        float
            Recovered numeric value.
        """
        if self.basis == "canonical":
            # Direct indexing: no dot product accumulation
            return float(embedding[0] * self.scale)
        else:
            projection = jnp.dot(embedding, self.direction)
            return float(projection * self.scale)

    def decode_batch(self, embeddings: jax.Array) -> jax.Array:
        """
        Recover numbers from a batch of embeddings.

        Parameters
        ----------
        embeddings : jax.Array
            Embeddings, shape [..., dim].

        Returns
        -------
        jax.Array
            Recovered values, shape [...].
        """
        if self.basis == "canonical":
            return embeddings[..., 0] * self.scale
        else:
            projection = jnp.sum(embeddings * self.direction, axis=-1)
            return projection * self.scale


def parse_arithmetic_expression(input_bytes: jax.Array) -> Tuple[jax.Array, int, jax.Array]:
    """
    Parse an arithmetic expression like "42+58=" into components.

    Parameters
    ----------
    input_bytes : jax.Array
        Byte sequence representing expression, shape [max_len].

    Returns
    -------
    operand1_bytes : jax.Array
        Bytes for first operand.
    operator : int
        Operator code (ord('+'), ord('-'), ord('*'), ord('/'))
    operand2_bytes : jax.Array
        Bytes for second operand.
    """
    PLUS = ord('+')
    MINUS = ord('-')
    STAR = ord('*')
    SLASH = ord('/')
    EQUALS = ord('=')
    ZERO = ord('0')
    NINE = ord('9')

    max_len = input_bytes.shape[0]
    positions = jnp.arange(max_len)

    is_plus = input_bytes == PLUS
    is_star = input_bytes == STAR
    is_slash = input_bytes == SLASH
    is_operator_char = is_plus | is_star | is_slash

    is_digit = (input_bytes >= ZERO) & (input_bytes <= NINE)
    prev_is_digit = jnp.concatenate([jnp.array([False]), is_digit[:-1]])
    is_binary_minus = (input_bytes == MINUS) & prev_is_digit
    is_operator = is_plus | is_star | is_slash | is_binary_minus

    operator_mask = is_operator & (positions > 0)
    operator_positions = jnp.where(operator_mask, positions, max_len)
    op_pos = jnp.min(operator_positions)

    operator = input_bytes[op_pos]

    equals_positions = jnp.where(input_bytes == EQUALS, positions, max_len)
    eq_pos = jnp.min(equals_positions)

    operand1_bytes = jnp.where(positions < op_pos, input_bytes, 0)

    operand2_bytes = jnp.where(
        (positions > op_pos) & (positions < eq_pos),
        input_bytes,
        0
    )
    op2_start = op_pos + 1
    operand2_bytes = jnp.roll(operand2_bytes, -int(op2_start))
    operand2_bytes = jnp.where(positions < (eq_pos - op_pos - 1), operand2_bytes, 0)

    return operand1_bytes, operator, operand2_bytes


def verify_linear_property(encoder: NumberEncoder, a: float, b: float, atol: float = 1e-6) -> bool:
    """
    Verify that encode(a) + encode(b) == encode(a+b).

    Parameters
    ----------
    encoder : NumberEncoder
        The encoder.
    a, b : float
        Numbers to test.
    atol : float
        Tolerance.

    Returns
    -------
    bool
        True if the linear property holds.
    """
    emb_a = encoder.encode_number(a)
    emb_b = encoder.encode_number(b)
    emb_sum = encoder.encode_number(a + b)

    computed_sum = emb_a + emb_b

    return bool(jnp.allclose(computed_sum, emb_sum, atol=atol))


if __name__ == "__main__":
    print("NumberEncoder demo")

    print("\n=== Canonical basis (default) ===")
    encoder = NumberEncoder(dim=256, scale=100000.0, basis="canonical")

    test_pairs = [
        (42, 58),
        (1000, 2000),
        (12345, 54321),
        (-100, 100),
        (99999, 1),
    ]

    print("\nLinearity verification:")

    for a, b in test_pairs:
        emb_a = encoder.encode_number(a)
        emb_b = encoder.encode_number(b)
        emb_sum = encoder.encode_number(a + b)
        computed = emb_a + emb_b

        error = float(jnp.linalg.norm(computed - emb_sum))
        passed = error < 1e-10

        print(f"  {a} + {b} = {a + b}")
        print(f"    ||encode(a) + encode(b) - encode(a+b)|| = {error:.2e}")
        print(f"    status: {'pass' if passed else 'fail'}")

    print("\nRound-trip (encode -> decode):")

    test_numbers = [0, 1, -1, 42, 1000, 99999, -12345]
    for n in test_numbers:
        emb = encoder.encode_number(n)
        recovered = encoder.decode(emb)
        error = abs(recovered - n)
        print(f"  {n:>7} -> encode -> decode -> {recovered:>10.2f} (error: {error:.2e})")

    print("\n=== Random orthonormal basis (old behavior) ===")
    encoder_old = NumberEncoder(dim=256, scale=100000.0, basis="random_orthonormal", seed=42)

    print("\nRound-trip (encode -> decode):")
    for n in test_numbers:
        emb = encoder_old.encode_number(n)
        recovered = encoder_old.decode(emb)
        error = abs(recovered - n)
        print(f"  {n:>7} -> encode -> decode -> {recovered:>10.2f} (error: {error:.2e})")
