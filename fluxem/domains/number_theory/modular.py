"""
Modular arithmetic for FluxEM-Domains.

Provides deterministic operations on integers mod n, including ModularInt class.
"""

from dataclasses import dataclass
from typing import Any, Optional
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
)

from .divisibility import extended_gcd


@dataclass(frozen=True)
class ModularInt:
    """
    Integer in Z/nZ (integers mod n).

    Value is always in range [0, modulus).
    All operations return new ModularInt with same modulus.
    """

    value: int
    modulus: int

    def __post_init__(self):
        """Ensure value is in valid range [0, modulus)."""
        if self.modulus <= 0:
            raise ValueError(f"modulus must be > 0, got {self.modulus}")

        # Normalize value to [0, modulus)
        normalized = self.value % self.modulus
        object.__setattr__(self, "value", normalized)

    def __add__(self, other: "ModularInt") -> "ModularInt":
        """Addition modulo n."""
        if self.modulus != other.modulus:
            raise ValueError(f"Moduli must match: {self.modulus} vs {other.modulus}")
        return ModularInt(self.value + other.value, self.modulus)

    def __sub__(self, other: "ModularInt") -> "ModularInt":
        """Subtraction modulo n."""
        if self.modulus != other.modulus:
            raise ValueError(f"Moduli must match: {self.modulus} vs {other.modulus}")
        return ModularInt(self.value - other.value, self.modulus)

    def __mul__(self, other: "ModularInt") -> "ModularInt":
        """Multiplication modulo n."""
        if self.modulus != other.modulus:
            raise ValueError(f"Moduli must match: {self.modulus} vs {other.modulus}")
        return ModularInt(self.value * other.value, self.modulus)

    def __pow__(self, exp: int) -> "ModularInt":
        """
        Exponentiation via binary exponentiation.

        Computes a^exp mod n efficiently.
        """
        if exp < 0:
            inv = self.inverse()
            if inv is None:
                raise ValueError(f"{self.value} has no inverse mod {self.modulus}")
            return inv ** (-exp)

        result = ModularInt(1, self.modulus)
        base = ModularInt(self.value, self.modulus)

        while exp > 0:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1

        return result

    def inverse(self) -> Optional["ModularInt"]:
        """
        Modular inverse using extended Euclidean algorithm.

        Returns ModularInt such that self * result = 1 (mod n).
        Returns None if inverse doesn't exist (not coprime with modulus).
        """
        g, x, _ = extended_gcd(self.value, self.modulus)

        if g != 1:
            return None

        inv = x % self.modulus
        return ModularInt(inv, self.modulus)

    def __eq__(self, other: object) -> bool:
        """Equality check."""
        if not isinstance(other, ModularInt):
            return False
        return self.value == other.value and self.modulus == other.modulus

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.value, self.modulus))

    def __repr__(self) -> str:
        return f"ModularInt({self.value}, {self.modulus})"


def mod_add(a: int, b: int, m: int) -> int:
    """
    (a + b) mod m.

    Returns the sum of a and b, reduced modulo m.
    """
    if m <= 0:
        raise ValueError(f"modulus must be > 0, got {m}")
    return (a + b) % m


def mod_mul(a: int, b: int, m: int) -> int:
    """
    (a * b) mod m.

    Returns the product of a and b, reduced modulo m.
    """
    if m <= 0:
        raise ValueError(f"modulus must be > 0, got {m}")
    return (a * b) % m


def mod_pow(base: int, exp: int, m: int) -> int:
    """
    (base ^ exp) mod m via binary exponentiation.

    Computes base^exp mod m efficiently using binary exponentiation.
    Handles negative exponents by computing modular inverse.
    """
    if m <= 0:
        raise ValueError(f"modulus must be > 0, got {m}")
    if m == 1:
        return 0

    if exp < 0:
        # Compute inverse and use positive exponent
        g, x, _ = extended_gcd(base, m)
        if g != 1:
            raise ValueError(f"{base} has no inverse mod {m}")
        base = x % m
        exp = -exp

    result = 1
    base = base % m

    while exp > 0:
        if exp & 1:
            result = mod_mul(result, base, m)
        base = mod_mul(base, base, m)
        exp >>= 1

    return result


def mod_inverse(a: int, m: int) -> Optional[int]:
    """
    Modular inverse of a mod m, or None if not coprime.

    Returns x such that a * x ≡ 1 (mod m).
    Returns None if no inverse exists (gcd(a, m) != 1).
    """
    if m <= 0:
        raise ValueError(f"modulus must be > 0, got {m}")

    g, x, _ = extended_gcd(a, m)

    if g != 1:
        return None

    return x % m


class ModularEncoder:
    """
    Encoder for modular integers.

    Embedding layout (domain-specific dims 8-71):
    - dim 8: value (mod modulus)
    - dim 9: modulus
    - dim 10-11: log encoding of modulus for large moduli
    - dims 72-79: properties (has_inverse, is_zero, is_unit, etc.)
    """

    domain_tag = DOMAIN_TAGS["num_modular"]
    domain_name = "num_modular"

    VALUE_POS = 8
    MODULUS_POS = 9
    MODULUS_LOG_POS = 10
    HAS_INVERSE_POS = 72
    IS_ZERO_POS = 73
    IS_UNIT_POS = 74

    def __init__(self):
        """Initialize the modular encoder."""
        pass

    def encode(self, modular_int: ModularInt) -> Any:
        """
        Encode a ModularInt to a 128-dim embedding.
        """
        backend = get_backend()
        embedding = backend.zeros((EMBEDDING_DIM,))

        # Domain tag
        embedding = backend.at_add(embedding, slice(0, 8), self.domain_tag)

        # Value and modulus
        embedding = backend.at_add(embedding, self.VALUE_POS, float(modular_int.value))
        embedding = backend.at_add(embedding, self.MODULUS_POS, float(modular_int.modulus))

        # Log encoding of modulus for large values
        if modular_int.modulus > 0:
            import math

            embedding = backend.at_add(embedding, self.MODULUS_LOG_POS, 
                math.log(float(modular_int.modulus))
            )

        # Properties
        from .divisibility import is_coprime

        has_inv = is_coprime(modular_int.value, modular_int.modulus)
        embedding = backend.at_add(embedding, self.HAS_INVERSE_POS, 1.0 if has_inv else 0.0)
        embedding = backend.at_add(embedding, self.IS_ZERO_POS, 
            1.0 if modular_int.value == 0 else 0.0
        )
        embedding = backend.at_add(embedding, self.IS_UNIT_POS, 
            1.0 if modular_int.value == 1 else 0.0
        )

        return embedding

    def decode(self, embedding: Any) -> ModularInt:
        """Decode an embedding back to a ModularInt."""
        value = int(round(float(embedding[self.VALUE_POS])))
        modulus = int(round(float(embedding[self.MODULUS_POS])))
        return ModularInt(value, modulus)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is valid for this domain."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # Operations on embeddings

    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two modular integers.
        """
        m1 = self.decode(emb1)
        m2 = self.decode(emb2)
        return self.encode(m1 + m2)

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """
        Subtract two modular integers.
        """
        m1 = self.decode(emb1)
        m2 = self.decode(emb2)
        return self.encode(m1 - m2)

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two modular integers.
        """
        m1 = self.decode(emb1)
        m2 = self.decode(emb2)
        return self.encode(m1 * m2)

    def power(self, embedding: Any, exp: int) -> Any:
        """
        Raise modular integer to a power.
        """
        m = self.decode(embedding)
        return self.encode(m**exp)

    def inverse(self, embedding: Any) -> Optional[Any]:
        """
        Compute modular inverse.
        Returns None if inverse doesn't exist.
        """
        m = self.decode(embedding)
        inv = m.inverse()
        if inv is None:
            return None
        return self.encode(inv)

    # Queries

    def has_inverse(self, embedding: Any) -> bool:
        """Check if modular integer has an inverse."""
        return float(embedding[self.HAS_INVERSE_POS]) > 0.5

    def is_zero(self, embedding: Any) -> bool:
        """Check if modular integer is zero."""
        return float(embedding[self.IS_ZERO_POS]) > 0.5

    def is_unit(self, embedding: Any) -> bool:
        """Check if modular integer is 1 (the multiplicative identity)."""
        return float(embedding[self.IS_UNIT_POS]) > 0.5

    def equals(self, emb1: Any, emb2: Any) -> bool:
        """Check if two embeddings represent the same modular integer."""
        m1 = self.decode(emb1)
        m2 = self.decode(emb2)
        return m1.value == m2.value and m1.modulus == m2.modulus
