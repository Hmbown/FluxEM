"""
Integer numbers and encoder for FluxEM-Domains.

Provides integer arithmetic with embeddings that preserve core properties.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import math
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    log_encode_value,
    log_decode_value,
)


@dataclass(frozen=True)
class Integer:
    """
    Immutable integer with arbitrary-precision arithmetic.

    Uses Python's arbitrary precision for computation.
    All operations return new Integer instances.
    """

    value: int

    def __post_init__(self):
        """Ensure value is actually an integer."""
        if not isinstance(self.value, int):
            raise ValueError(f"Integer.value must be int, got {type(self.value)}")

    def __add__(self, other: "Integer") -> "Integer":
        """Addition. Python integer arithmetic."""
        return Integer(self.value + other.value)

    def __sub__(self, other: "Integer") -> "Integer":
        """Subtraction. Python integer arithmetic."""
        return Integer(self.value - other.value)

    def __mul__(self, other: "Integer") -> "Integer":
        """Multiplication. Python integer arithmetic."""
        return Integer(self.value * other.value)

    def __floordiv__(self, other: "Integer") -> "Integer":
        """Floor division. Python integer arithmetic."""
        if other.value == 0:
            raise ZeroDivisionError("division by zero")
        return Integer(self.value // other.value)

    def __mod__(self, other: "Integer") -> "Integer":
        """Modulo. Python integer arithmetic."""
        if other.value == 0:
            raise ZeroDivisionError("modulo by zero")
        return Integer(self.value % other.value)

    def __neg__(self) -> "Integer":
        """Negation. Python integer arithmetic."""
        return Integer(-self.value)

    def __abs__(self) -> "Integer":
        """Absolute value. Python integer arithmetic."""
        return Integer(abs(self.value))

    def __eq__(self, other: object) -> bool:
        """Equality check."""
        if not isinstance(other, Integer):
            return False
        return self.value == other.value

    def __lt__(self, other: "Integer") -> bool:
        """Less than. Python integer comparison."""
        return self.value < other.value

    def __le__(self, other: "Integer") -> bool:
        """Less than or equal. Python integer comparison."""
        return self.value <= other.value

    def __gt__(self, other: "Integer") -> bool:
        """Greater than. Python integer comparison."""
        return self.value > other.value

    def __ge__(self, other: "Integer") -> bool:
        """Greater than or equal. Python integer comparison."""
        return self.value >= other.value

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash(self.value)

    @property
    def sign(self) -> int:
        """Get sign: -1, 0, or 1."""
        if self.value > 0:
            return 1
        elif self.value < 0:
            return -1
        return 0

    @property
    def is_positive(self) -> bool:
        """Check if positive."""
        return self.value > 0

    @property
    def is_negative(self) -> bool:
        """Check if negative."""
        return self.value < 0

    @property
    def is_zero(self) -> bool:
        """Check if zero."""
        return self.value == 0

    @property
    def is_even(self) -> bool:
        """Check if even. value % 2 == 0."""
        return self.value % 2 == 0

    @property
    def is_odd(self) -> bool:
        """Check if odd. value % 2 == 1."""
        return self.value % 2 != 0

    @property
    def is_power_of_2(self) -> bool:
        """Check if power of 2. bit count == 1 for positive."""
        if self.value <= 0:
            return False
        return (self.value & (self.value - 1)) == 0

    @property
    def bit_length(self) -> int:
        """Number of bits needed to represent."""
        return self.value.bit_length()

    def __repr__(self) -> str:
        return f"Integer({self.value})"

    def __str__(self) -> str:
        return str(self.value)


class IntegerEncoder:
    """
    Encoder for integers.

    Embedding layout (domain-specific dims 8-71):
    - dim 8:  sign (-1, 0, or 1)
    - dim 9:  log10(|value|) for magnitude
    - dims 10-17: low bits of value for small integers (direct recovery)
    - dims 72-79: properties (is_positive, is_even, is_prime, is_power_of_2, etc.)
    """

    domain_tag = DOMAIN_TAGS["num_integer"]
    domain_name = "num_integer"

    SIGN_POS = 8
    LOG_MAG_POS = 9
    LOW_BITS_START = 10
    LOW_BITS_COUNT = 8

    IS_POSITIVE_POS = 72
    IS_EVEN_POS = 73
    IS_PRIME_POS = 74
    IS_POWER_OF_2_POS = 75
    BIT_COUNT_POS = 76

    def __init__(self):
        """Initialize the integer encoder."""
        pass

    def encode(self, integer: Integer) -> Any:
        """
        Encode an Integer to a 128-dim embedding.

        Uses log-magnitude representation for large numbers,
        stores low bits for direct recovery of small integers.
        """
        backend = get_backend()
        from .primes import is_prime

        embedding = backend.zeros((EMBEDDING_DIM,))

        # Domain tag
        embedding = backend.at_add(embedding, slice(0, 8), self.domain_tag)

        # Sign
        embedding = backend.at_add(embedding, self.SIGN_POS, float(integer.sign))

        # Log magnitude
        if integer.value == 0:
            embedding = backend.at_add(embedding, self.LOG_MAG_POS, -100.0)
        else:
            sign, log_mag = log_encode_value(float(integer.value))
            embedding = backend.at_add(embedding, self.LOG_MAG_POS, log_mag)

        # Low bits for direct recovery of small integers
        for i in range(self.LOW_BITS_COUNT):
            if (integer.value >> i) & 1:
                embedding = backend.at_add(embedding, self.LOW_BITS_START + i, 1.0)

        # Properties
        embedding = backend.at_add(embedding, self.IS_POSITIVE_POS, 
            1.0 if integer.is_positive else 0.0
        )
        embedding = backend.at_add(embedding, self.IS_EVEN_POS, 1.0 if integer.is_even else 0.0)
        embedding = backend.at_add(embedding, self.IS_PRIME_POS, 
            1.0 if is_prime(integer.value) else 0.0
        )
        embedding = backend.at_add(embedding, self.IS_POWER_OF_2_POS, 
            1.0 if integer.is_power_of_2 else 0.0
        )
        embedding = backend.at_add(embedding, self.BIT_COUNT_POS, float(integer.bit_length))

        return embedding

    def decode(self, embedding: Any) -> Integer:
        """
        Decode an embedding back to an Integer.

        Uses log magnitude for recovery.
        Low bits are only used for very small positive integers.
        """
        sign = float(embedding[self.SIGN_POS])
        log_mag = float(embedding[self.LOG_MAG_POS])

        # Use log magnitude for accurate decoding
        decoded = log_decode_value(sign, log_mag)
        result = int(round(decoded))

        return Integer(result)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is valid for this domain."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # Integer operations on embeddings

    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two integers.

        Decode, add, re-encode.
        """
        n1 = self.decode(emb1)
        n2 = self.decode(emb2)
        return self.encode(n1 + n2)

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """
        Subtract two integers.

        Decode, subtract, re-encode.
        """
        n1 = self.decode(emb1)
        n2 = self.decode(emb2)
        return self.encode(n1 - n2)

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two integers.

        Decode, multiply, re-encode.
        """
        n1 = self.decode(emb1)
        n2 = self.decode(emb2)
        return self.encode(n1 * n2)

    def floor_divide(self, emb1: Any, emb2: Any) -> Any:
        """
        Floor divide two integers.

        Decode, divide, re-encode.
        """
        n1 = self.decode(emb1)
        n2 = self.decode(emb2)
        return self.encode(n1 // n2)

    def mod(self, emb1: Any, emb2: Any) -> Any:
        """
        Compute modulo of two integers.

        Decode, mod, re-encode.
        """
        n1 = self.decode(emb1)
        n2 = self.decode(emb2)
        return self.encode(n1 % n2)

    def negate(self, embedding: Any) -> Any:
        """
        Negate an integer.

        Decode, negate, re-encode.
        """
        n = self.decode(embedding)
        return self.encode(-n)

    def absolute(self, embedding: Any) -> Any:
        """
        Absolute value of integer.

        Decode, abs, re-encode.
        """
        n = self.decode(embedding)
        return self.encode(abs(n))

    # === QUERIES ===

    def is_positive(self, embedding: Any) -> bool:
        """Check if integer is positive."""
        return float(embedding[self.IS_POSITIVE_POS]) > 0.5

    def is_even(self, embedding: Any) -> bool:
        """Check if integer is even."""
        return float(embedding[self.IS_EVEN_POS]) > 0.5

    def is_prime(self, embedding: Any) -> bool:
        """Check if integer is prime."""
        return float(embedding[self.IS_PRIME_POS]) > 0.5

    def is_power_of_2(self, embedding: Any) -> bool:
        """Check if integer is power of 2."""
        return float(embedding[self.IS_POWER_OF_2_POS]) > 0.5

    def compare(self, emb1: Any, emb2: Any) -> int:
        """
        Compare two integers.

        Returns: -1 if n1 < n2, 0 if equal, 1 if n1 > n2.
        Decode and compare.
        """
        n1 = self.decode(emb1)
        n2 = self.decode(emb2)
        if n1 < n2:
            return -1
        elif n1 > n2:
            return 1
        return 0

    def equals(self, emb1: Any, emb2: Any) -> bool:
        """Check if two embeddings represent the same integer."""
        return self.compare(emb1, emb2) == 0
