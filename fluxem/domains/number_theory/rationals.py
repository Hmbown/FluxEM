"""
Rational numbers for FluxEM-Domains.

Provides rational numbers as numerator/denominator with proper normalization.
"""

from typing import Any

from dataclasses import dataclass
import math
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    log_encode_value,
    log_decode_value,
)

from .divisibility import gcd


@dataclass(frozen=True)
class Rational:
    """
    Rational number as numerator/denominator.

    Denominator is always positive and in lowest terms.
    All operations return new Rational instances.
    """

    numerator: int
    denominator: int

    def __post_init__(self):
        """Normalize: reduce to lowest terms, ensure denominator > 0."""
        if self.denominator == 0:
            raise ValueError("Denominator cannot be zero")

        num = self.numerator
        den = self.denominator

        # Ensure denominator is positive
        if den < 0:
            num = -num
            den = -den

        # Reduce to lowest terms
        g = gcd(abs(num), den)
        if g > 1:
            num = num // g
            den = den // g

        object.__setattr__(self, "numerator", num)
        object.__setattr__(self, "denominator", den)

    def __add__(self, other: "Rational") -> "Rational":
        """
        Addition. (a/b) + (c/d) = (ad + bc) / bd.
        """
        if self.denominator == other.denominator:
            return Rational(self.numerator + other.numerator, self.denominator)

        num = self.numerator * other.denominator + other.numerator * self.denominator
        den = self.denominator * other.denominator
        return Rational(num, den)

    def __sub__(self, other: "Rational") -> "Rational":
        """
        Subtraction. (a/b) - (c/d) = (ad - bc) / bd.
        """
        if self.denominator == other.denominator:
            return Rational(self.numerator - other.numerator, self.denominator)

        num = self.numerator * other.denominator - other.numerator * self.denominator
        den = self.denominator * other.denominator
        return Rational(num, den)

    def __mul__(self, other: "Rational") -> "Rational":
        """
        Multiplication. (a/b) * (c/d) = (ac) / (bd).
        """
        num = self.numerator * other.numerator
        den = self.denominator * other.denominator
        return Rational(num, den)

    def __truediv__(self, other: "Rational") -> "Rational":
        """
        Division. (a/b) / (c/d) = (ad) / (bc).
        """
        if other.numerator == 0:
            raise ZeroDivisionError("division by zero")

        num = self.numerator * other.denominator
        den = self.denominator * other.numerator
        return Rational(num, den)

    def __neg__(self) -> "Rational":
        """Negation. -(a/b) = (-a)/b."""
        return Rational(-self.numerator, self.denominator)

    def __abs__(self) -> "Rational":
        """Absolute value."""
        return Rational(abs(self.numerator), self.denominator)

    def __eq__(self, other: object) -> bool:
        """
        Equality check via cross-multiplication.

        a/b == c/d iff ad == bc
        """
        if not isinstance(other, Rational):
            return False
        return self.numerator * other.denominator == other.numerator * self.denominator

    def __lt__(self, other: "Rational") -> bool:
        """
        Less than via cross-multiplication.

        a/b < c/d iff ad < bc (when b,d > 0)
        """
        return self.numerator * other.denominator < other.numerator * self.denominator

    def __le__(self, other: "Rational") -> bool:
        """Less than or equal."""
        return self.numerator * other.denominator <= other.numerator * self.denominator

    def __gt__(self, other: "Rational") -> bool:
        """Greater than."""
        return self.numerator * other.denominator > other.numerator * self.denominator

    def __ge__(self, other: "Rational") -> bool:
        """Greater than or equal."""
        return self.numerator * other.denominator >= other.numerator * self.denominator

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.numerator, self.denominator))

    def __float__(self) -> float:
        """Convert to float (approximate)."""
        return self.numerator / self.denominator

    def __int__(self) -> int:
        """Convert to int (floor)."""
        return self.numerator // self.denominator

    def floor(self) -> int:
        """Floor value."""
        return int(self)

    def ceil(self) -> int:
        """Ceiling value."""
        return -((-self.numerator) // self.denominator)

    def is_integer(self) -> bool:
        """Check if rational is an integer (denominator == 1)."""
        return self.denominator == 1

    def is_positive(self) -> bool:
        """Check if positive."""
        return self.numerator > 0

    def is_negative(self) -> bool:
        """Check if negative."""
        return self.numerator < 0

    def is_zero(self) -> bool:
        """Check if zero."""
        return self.numerator == 0

    def to_float(self) -> float:
        """Convert to float."""
        return float(self)

    def __repr__(self) -> str:
        return f"Rational({self.numerator}, {self.denominator})"

    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"


class RationalEncoder:
    """
    Encoder for rational numbers.

    Embedding layout (domain-specific dims 8-71):
    - dims 8-9: numerator (sign, log_mag)
    - dims 10-11: denominator (sign=1, log_mag)
    - dim 12: sign of rational
    - dim 13: is_integer flag
    - dim 14: is_positive flag
    - dim 72: reduced form indicator
    """

    domain_tag = DOMAIN_TAGS["num_rational"]
    domain_name = "num_rational"

    NUM_SIGN_POS = 8
    NUM_LOG_POS = 9
    DEN_LOG_POS = 10
    DEN_SIGN_POS = 11
    RATIONAL_SIGN_POS = 12
    IS_INTEGER_POS = 13
    IS_POSITIVE_POS = 14
    REDUCED_POS = 72

    def __init__(self):
        """Initialize rational encoder."""
        pass

    def encode(self, rational: Rational) -> Any:
        """
        Encode a Rational to a 128-dim embedding.

        Uses log-magnitude representation for scale invariance.
        Stores both numerator and denominator exactly.
        """
        backend = get_backend()
        embedding = backend.zeros((EMBEDDING_DIM,))

        # Domain tag
        embedding = backend.at_add(embedding, slice(0, 8), self.domain_tag)

        # Numerator encoding
        if rational.numerator == 0:
            embedding = backend.at_add(embedding, self.NUM_SIGN_POS, 0.0)
            embedding = backend.at_add(embedding, self.NUM_LOG_POS, -100.0)
        else:
            num_sign, num_log = log_encode_value(float(rational.numerator))
            embedding = backend.at_add(embedding, self.NUM_SIGN_POS, num_sign)
            embedding = backend.at_add(embedding, self.NUM_LOG_POS, num_log)

        # Denominator encoding (always positive)
        den_sign, den_log = log_encode_value(float(rational.denominator))
        embedding = backend.at_add(embedding, self.DEN_SIGN_POS, den_sign)
        embedding = backend.at_add(embedding, self.DEN_LOG_POS, den_log)

        # Rational sign
        if rational.numerator > 0:
            embedding = backend.at_add(embedding, self.RATIONAL_SIGN_POS, 1.0)
        elif rational.numerator < 0:
            embedding = backend.at_add(embedding, self.RATIONAL_SIGN_POS, -1.0)
        else:
            embedding = backend.at_add(embedding, self.RATIONAL_SIGN_POS, 0.0)

        # Properties
        embedding = backend.at_add(embedding, self.IS_INTEGER_POS, 
            1.0 if rational.is_integer() else 0.0
        )
        embedding = backend.at_add(embedding, self.IS_POSITIVE_POS, 
            1.0 if rational.is_positive() else 0.0
        )

        # Reduced form indicator (always 1.0 since Rational always in reduced form)
        embedding = backend.at_add(embedding, self.REDUCED_POS, 1.0)

        return embedding

    def decode(self, embedding: Any) -> Rational:
        """
        Decode an embedding back to a Rational.

        Recovers numerator and denominator from log encoding.
        """
        # Decode numerator
        num_sign = float(embedding[self.NUM_SIGN_POS])
        num_log = float(embedding[self.NUM_LOG_POS])

        if abs(num_sign) < 0.5:
            numerator = 0
        else:
            numerator = int(round(log_decode_value(num_sign, num_log)))

        # Decode denominator
        den_log = float(embedding[self.DEN_LOG_POS])
        denominator = int(round(math.exp(den_log)))
        if denominator < 1:
            denominator = 1

        return Rational(numerator, denominator)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is valid for this domain."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # === OPERATIONS ON EMBEDDINGS ===

    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two rationals.

        Decode, add, re-encode.
        """
        r1 = self.decode(emb1)
        r2 = self.decode(emb2)
        return self.encode(r1 + r2)

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """
        Subtract two rationals.

        Decode, subtract, re-encode.
        """
        r1 = self.decode(emb1)
        r2 = self.decode(emb2)
        return self.encode(r1 - r2)

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two rationals.

        Decode, multiply, re-encode.
        """
        r1 = self.decode(emb1)
        r2 = self.decode(emb2)
        return self.encode(r1 * r2)

    def divide(self, emb1: Any, emb2: Any) -> Any:
        """
        Divide two rationals.

        Decode, divide, re-encode.
        """
        r1 = self.decode(emb1)
        r2 = self.decode(emb2)
        return self.encode(r1 / r2)

    def negate(self, embedding: Any) -> Any:
        """
        Negate a rational.

        Decode, negate, re-encode.
        """
        r = self.decode(embedding)
        return self.encode(-r)

    def absolute(self, embedding: Any) -> Any:
        """
        Absolute value of rational.

        Decode, abs, re-encode.
        """
        r = self.decode(embedding)
        return self.encode(abs(r))

    # === QUERIES ===

    def is_integer(self, embedding: Any) -> bool:
        """Check if rational is an integer."""
        return float(embedding[self.IS_INTEGER_POS]) > 0.5

    def is_positive(self, embedding: Any) -> bool:
        """Check if rational is positive."""
        return float(embedding[self.IS_POSITIVE_POS]) > 0.5

    def is_negative(self, embedding: Any) -> bool:
        """Check if rational is negative."""
        rational_sign = float(embedding[self.RATIONAL_SIGN_POS])
        return rational_sign < -0.5

    def is_zero(self, embedding: Any) -> bool:
        """Check if rational is zero."""
        return float(embedding[self.NUM_SIGN_POS]) == 0.0

    def compare(self, emb1: Any, emb2: Any) -> int:
        """
        Compare two rationals.

        Returns: -1 if r1 < r2, 0 if equal, 1 if r1 > r2.
        Decode and compare.
        """
        r1 = self.decode(emb1)
        r2 = self.decode(emb2)
        if r1 < r2:
            return -1
        elif r1 > r2:
            return 1
        return 0

    def equals(self, emb1: Any, emb2: Any) -> bool:
        """Check if two embeddings represent the same rational."""
        return self.compare(emb1, emb2) == 0
