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

from typing import Tuple, Optional, Literal, Any

from ..backend import get_backend


class NumberEncoder:
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

    direction: Any  # Unit direction vector [dim]
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

        backend = get_backend()

        if basis == "canonical":
            # e0: first coordinate only
            direction = backend.zeros(dim)
            direction = backend.at_set(direction, 0, 1.0)
            self.direction = direction
        else:
            # Random unit vector (old behavior)
            direction = backend.random_normal(dim, seed=seed)
            self.direction = direction / backend.norm(direction)

    def encode_number(self, n: float) -> Any:
        """
        Encode a number to a linear embedding.

        Parameters
        ----------
        n : float
            The numeric value.

        Returns
        -------
        Array
            Linear embedding, shape [dim].
        """
        backend = get_backend()

        if self.basis == "canonical":
            # Direct indexing: x[0] = n/scale, rest zeros
            emb = backend.zeros(self.dim)
            return backend.at_set(emb, 0, n / self.scale)
        else:
            return (n / self.scale) * self.direction

    def encode_string(self, digit_str: str) -> Any:
        """
        Encode a digit string to a linear embedding.

        Parameters
        ----------
        digit_str : str
            String representation of a number (e.g., "42", "-123", "3.14").

        Returns
        -------
        Array
            Linear embedding, shape [dim].
        """
        try:
            value = float(digit_str.strip())
        except (ValueError, AttributeError):
            value = 0.0
        return self.encode_number(value)

    def encode_bytes(self, byte_seq: Any) -> Any:
        """
        Encode a byte sequence representing a number.

        Parameters
        ----------
        byte_seq : Array
            Byte sequence (ASCII values), shape [max_len].
            Padded with zeros.

        Returns
        -------
        Array
            Linear embedding, shape [dim].
        """
        backend = get_backend()
        value = self._parse_number_from_bytes(byte_seq)
        if self.basis == "canonical":
            emb = backend.zeros(self.dim)
            return backend.at_set(emb, 0, value / self.scale)
        else:
            return (value / self.scale) * self.direction

    def _parse_number_from_bytes(self, byte_seq: Any) -> Any:
        """
        Parse a number from a byte sequence using backend operations.

        This implements place-value parsing:
        "123" = 1*100 + 2*10 + 3*1

        Handles negative numbers and ignores non-digit characters.
        """
        backend = get_backend()

        ZERO = ord('0')
        NINE = ord('9')
        MINUS = ord('-')

        is_digit = (byte_seq >= ZERO) & (byte_seq <= NINE)
        is_minus = byte_seq == MINUS

        first_nonzero_idx = backend.argmax(byte_seq > 0)
        is_negative = byte_seq[first_nonzero_idx] == MINUS

        digit_values = backend.where(is_digit, byte_seq - ZERO, backend.zeros(len(byte_seq)))

        positions = backend.arange(len(byte_seq))
        valid_positions = backend.where(is_digit, positions, len(byte_seq))
        first_digit_pos = backend.argmin(valid_positions)
        valid_positions_for_max = backend.where(is_digit, positions, -1)
        last_digit_pos = backend.argmax(valid_positions_for_max)

        n_digits = backend.sum(is_digit.astype(float) if hasattr(is_digit, 'astype') else is_digit)

        # Cumulative sum for position tracking
        cumsum = []
        total = 0
        for i in range(len(byte_seq)):
            if is_digit[i]:
                total += 1
            cumsum.append(total)
        position_in_number = backend.array(cumsum) - 1

        place_values = backend.where(
            is_digit,
            backend.power(10.0, n_digits - 1 - position_in_number),
            backend.zeros(len(byte_seq))
        )

        value = backend.sum(digit_values * place_values)
        value = backend.where(is_negative, -value, value)

        return value

    def decode(self, embedding: Any) -> float:
        """
        Recover the number from a linear embedding.

        Parameters
        ----------
        embedding : Array
            Linear embedding, shape [dim].

        Returns
        -------
        float
            Recovered numeric value.
        """
        backend = get_backend()

        if self.basis == "canonical":
            # Direct indexing: no dot product accumulation
            return float(embedding[0] * self.scale)
        else:
            projection = backend.dot(embedding, self.direction)
            return float(projection * self.scale)

    def decode_batch(self, embeddings: Any) -> Any:
        """
        Recover numbers from a batch of embeddings.

        Parameters
        ----------
        embeddings : Array
            Embeddings, shape [..., dim].

        Returns
        -------
        Array
            Recovered values, shape [...].
        """
        backend = get_backend()

        if self.basis == "canonical":
            return embeddings[..., 0] * self.scale
        else:
            projection = backend.sum(embeddings * self.direction, axis=-1)
            return projection * self.scale


def parse_arithmetic_expression(input_bytes: Any) -> Tuple[Any, int, Any]:
    """
    Parse an arithmetic expression like "42+58=" into components.

    Parameters
    ----------
    input_bytes : Array
        Byte sequence representing expression, shape [max_len].

    Returns
    -------
    operand1_bytes : Array
        Bytes for first operand.
    operator : int
        Operator code (ord('+'), ord('-'), ord('*'), ord('/'))
    operand2_bytes : Array
        Bytes for second operand.
    """
    backend = get_backend()

    PLUS = ord('+')
    MINUS = ord('-')
    STAR = ord('*')
    SLASH = ord('/')
    EQUALS = ord('=')
    ZERO = ord('0')
    NINE = ord('9')

    max_len = input_bytes.shape[0]
    positions = backend.arange(max_len)

    is_plus = input_bytes == PLUS
    is_star = input_bytes == STAR
    is_slash = input_bytes == SLASH
    is_operator_char = is_plus | is_star | is_slash

    is_digit = (input_bytes >= ZERO) & (input_bytes <= NINE)
    # Handle prev_is_digit by padding
    prev_is_digit = backend.concatenate([backend.array([False]), is_digit[:-1]])
    is_binary_minus = (input_bytes == MINUS) & prev_is_digit
    is_operator = is_plus | is_star | is_slash | is_binary_minus

    operator_mask = is_operator & (positions > 0)
    operator_positions = backend.where(operator_mask, positions, max_len)
    op_pos = int(backend.argmin(operator_positions))

    operator = int(input_bytes[op_pos])

    equals_positions = backend.where(input_bytes == EQUALS, positions, max_len)
    eq_pos = int(backend.argmin(equals_positions))

    operand1_bytes = backend.where(positions < op_pos, input_bytes, 0)

    operand2_bytes = backend.where(
        (positions > op_pos) & (positions < eq_pos),
        input_bytes,
        0
    )

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
    backend = get_backend()

    emb_a = encoder.encode_number(a)
    emb_b = encoder.encode_number(b)
    emb_sum = encoder.encode_number(a + b)

    computed_sum = emb_a + emb_b

    return backend.allclose(computed_sum, emb_sum, atol=atol)


if __name__ == "__main__":
    print("NumberEncoder demo")

    print("\n=== Canonical basis (default) ===")
    encoder = NumberEncoder(dim=256, scale=100000.0, basis="canonical")

    backend = get_backend()
    print(f"Using backend: {backend.name}")

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

        error = float(backend.norm(computed - emb_sum))
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
