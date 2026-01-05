"""
Dimensional Analysis Encoder for Physics.

Embeds physical quantities with SI dimensions as vector exponents.
Operations on embeddings preserve dimensional correctness:
- Multiplication: Add dimension exponents
- Division: Subtract dimension exponents
- Addition: Only valid when dimensions match (type-checking)

Provides deterministic dimensional analysis without learned parameters.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from ...backend import get_backend
import math

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


# =============================================================================
# SI Base Dimensions
# =============================================================================

# The 7 SI base units - dimensions are exponent vectors over these
SI_DIMENSIONS = ['M', 'L', 'T', 'I', 'Theta', 'N', 'J']
# M = mass (kg)
# L = length (m)
# T = time (s)
# I = electric current (A)
# Theta = temperature (K)
# N = amount of substance (mol)
# J = luminous intensity (cd)

# Embedding layout within domain-specific region (64 dims starting at offset 8):
# dims 0-6:   SI exponents [M, L, T, I, Theta, N, J]
# dim 7:      Dimensionless flag (1.0 if all exponents are 0)
# dims 8-9:   Magnitude (sign, log|value|)
# dim 10:     Is zero flag
# dims 11-16: Reserved for derived unit hints
# dims 17-63: Reserved for future use

DIM_EXPONENT_OFFSET = 0  # Within domain-specific slice
DIM_EXPONENT_SIZE = 7
DIMENSIONLESS_FLAG = 7
MAGNITUDE_SIGN = 8
MAGNITUDE_LOG = 9
IS_ZERO_FLAG = 10
DERIVED_HINTS_OFFSET = 11


@dataclass(frozen=True)
class Dimensions:
    """
    Immutable representation of physical dimensions as SI exponents.

    Examples:
        Velocity: Dimensions(L=1, T=-1)        # m/s
        Force:    Dimensions(M=1, L=1, T=-2)   # kg⋅m/s² = N
        Energy:   Dimensions(M=1, L=2, T=-2)   # kg⋅m²/s² = J
    """
    M: int = 0      # mass (kg)
    L: int = 0      # length (m)
    T: int = 0      # time (s)
    I: int = 0      # current (A)
    Theta: int = 0  # temperature (K)
    N: int = 0      # amount (mol)
    J: int = 0      # luminous intensity (cd)

    def to_vector(self) -> Any:
        """Convert to 7-element exponent vector."""
        backend = get_backend()
        return backend.array([self.M, self.L, self.T, self.I, self.Theta, self.N, self.J],
                       dtype=None)

    @classmethod
    def from_vector(cls, vec: Any) -> 'Dimensions':
        """Create from 7-element exponent vector."""
        return cls(
            M=int(round(vec[0].item())),
            L=int(round(vec[1].item())),
            T=int(round(vec[2].item())),
            I=int(round(vec[3].item())),
            Theta=int(round(vec[4].item())),
            N=int(round(vec[5].item())),
            J=int(round(vec[6].item())),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'Dimensions':
        """Create from dictionary like {'L': 1, 'T': -1}."""
        return cls(**{k: d.get(k, 0) for k in ['M', 'L', 'T', 'I', 'Theta', 'N', 'J']})

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary, omitting zero exponents."""
        result = {}
        for name, val in [('M', self.M), ('L', self.L), ('T', self.T),
                          ('I', self.I), ('Theta', self.Theta),
                          ('N', self.N), ('J', self.J)]:
            if val != 0:
                result[name] = val
        return result

    def is_dimensionless(self) -> bool:
        """Check if this is dimensionless (all exponents zero)."""
        return all(v == 0 for v in [self.M, self.L, self.T, self.I,
                                     self.Theta, self.N, self.J])

    def __mul__(self, other: 'Dimensions') -> 'Dimensions':
        """Multiply dimensions: add exponents."""
        return Dimensions(
            M=self.M + other.M, L=self.L + other.L, T=self.T + other.T,
            I=self.I + other.I, Theta=self.Theta + other.Theta,
            N=self.N + other.N, J=self.J + other.J,
        )

    def __truediv__(self, other: 'Dimensions') -> 'Dimensions':
        """Divide dimensions: subtract exponents."""
        return Dimensions(
            M=self.M - other.M, L=self.L - other.L, T=self.T - other.T,
            I=self.I - other.I, Theta=self.Theta - other.Theta,
            N=self.N - other.N, J=self.J - other.J,
        )

    def __pow__(self, n: int) -> 'Dimensions':
        """Raise dimensions to integer power: multiply exponents."""
        return Dimensions(
            M=self.M * n, L=self.L * n, T=self.T * n,
            I=self.I * n, Theta=self.Theta * n,
            N=self.N * n, J=self.J * n,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimensions):
            return False
        return (self.M == other.M and self.L == other.L and self.T == other.T and
                self.I == other.I and self.Theta == other.Theta and
                self.N == other.N and self.J == other.J)

    def __repr__(self) -> str:
        parts = []
        for name, val in [('M', self.M), ('L', self.L), ('T', self.T),
                          ('I', self.I), ('Θ', self.Theta),
                          ('N', self.N), ('J', self.J)]:
            if val != 0:
                if val == 1:
                    parts.append(name)
                else:
                    parts.append(f"{name}^{val}")
        return f"[{' '.join(parts)}]" if parts else "[dimensionless]"


# =============================================================================
# Common Derived Dimensions (for hints)
# =============================================================================

DIMENSIONLESS = Dimensions()
METER = Dimensions(L=1)
SECOND = Dimensions(T=1)
KILOGRAM = Dimensions(M=1)
AMPERE = Dimensions(I=1)
KELVIN = Dimensions(Theta=1)
MOLE = Dimensions(N=1)
CANDELA = Dimensions(J=1)

# Derived SI units
VELOCITY = Dimensions(L=1, T=-1)           # m/s
ACCELERATION = Dimensions(L=1, T=-2)       # m/s²
FORCE = Dimensions(M=1, L=1, T=-2)         # N = kg⋅m/s²
ENERGY = Dimensions(M=1, L=2, T=-2)        # J = kg⋅m²/s²
POWER = Dimensions(M=1, L=2, T=-3)         # W = kg⋅m²/s³
PRESSURE = Dimensions(M=1, L=-1, T=-2)     # Pa = kg/(m⋅s²)
FREQUENCY = Dimensions(T=-1)               # Hz = 1/s
CHARGE = Dimensions(I=1, T=1)              # C = A⋅s
VOLTAGE = Dimensions(M=1, L=2, T=-3, I=-1) # V = kg⋅m²/(A⋅s³)
RESISTANCE = Dimensions(M=1, L=2, T=-3, I=-2)  # Ω


# =============================================================================
# Dimensional Quantity Encoder
# =============================================================================

class DimensionalQuantity:
    """
    Encoder for physical quantities with dimensions.

    Embedding structure (within 64-dim domain-specific region):
        dims 0-6:  SI exponents [M, L, T, I, Θ, N, J]
        dim 7:     Dimensionless flag
        dims 8-9:  Magnitude (sign, log|value|)
        dim 10:    Is zero flag
        dims 11-16: Derived unit hints

    All dimensional operations are deterministic given the encoded fields.
    """

    domain_tag = DOMAIN_TAGS["phys_quantity"]
    domain_name = "phys_quantity"

    def encode(self, value: float, dimensions: Union[Dimensions, Dict[str, int]]) -> Any:
        """
        Encode a physical quantity.

        Args:
            value: Numeric magnitude
            dimensions: Dimensions object or dict like {'L': 1, 'T': -1}

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(dimensions, dict):
            dimensions = Dimensions.from_dict(dimensions)

        emb = create_embedding()

        # Set domain tag (dims 0-7)
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Set dimension exponents (dims 8-14 in full embedding = 0-6 in specific)
        dim_vec = dimensions.to_vector()
        for i in range(7):
            emb = backend.at_add(emb, 8 + i, dim_vec[i])

        # Dimensionless flag
        emb = backend.at_add(emb, 8 + DIMENSIONLESS_FLAG, 
            1.0 if dimensions.is_dimensionless() else 0.0
        )

        # Magnitude encoding
        if abs(value) < EPSILON:
            emb = backend.at_add(emb, 8 + MAGNITUDE_SIGN, 0.0)
            emb = backend.at_add(emb, 8 + MAGNITUDE_LOG, -100.0)
            emb = backend.at_add(emb, 8 + IS_ZERO_FLAG, 1.0)
        else:
            sign, log_mag = log_encode_value(value)
            emb = backend.at_add(emb, 8 + MAGNITUDE_SIGN, sign)
            emb = backend.at_add(emb, 8 + MAGNITUDE_LOG, log_mag)
            emb = backend.at_add(emb, 8 + IS_ZERO_FLAG, 0.0)

        # Derived unit hints
        emb = self._set_derived_hints(emb, dimensions)

        return emb

    def decode(self, emb: Any) -> Tuple[float, Dimensions]:
        """
        Decode an embedding to (value, dimensions).

        Args:
            emb: 128-dim embedding

        Returns:
            Tuple of (magnitude, Dimensions)
        """
        # Extract dimensions
        dim_vec = emb[8:8 + 7]
        dimensions = Dimensions.from_vector(dim_vec)

        # Extract magnitude
        is_zero = emb[8 + IS_ZERO_FLAG].item() > 0.5
        if is_zero:
            return (0.0, dimensions)

        sign = emb[8 + MAGNITUDE_SIGN].item()
        log_mag = emb[8 + MAGNITUDE_LOG].item()
        value = log_decode_value(sign, log_mag)

        return (value, dimensions)

    def decode_value(self, emb: Any) -> float:
        """Decode just the magnitude."""
        value, _ = self.decode(emb)
        return value

    def decode_dimensions(self, emb: Any) -> Dimensions:
        """Decode just the dimensions."""
        _, dims = self.decode(emb)
        return dims

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid physical quantity."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Algebraic operations (deterministic)
    # =========================================================================

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two physical quantities.

        Dimensions: Add exponents
        Magnitude: Multiply (add log-magnitudes)

        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Add dimension exponents
        for i in range(7):
            result = backend.at_add(result, 8 + i, emb1[8 + i] + emb2[8 + i])

        # Update dimensionless flag
        is_dimensionless = all(
            abs(result[8 + i].item()) < 0.5 for i in range(7)
        )
        result = backend.at_add(result, 8 + DIMENSIONLESS_FLAG, 1.0 if is_dimensionless else 0.0)

        # Multiply magnitudes (add logs)
        is_zero1 = emb1[8 + IS_ZERO_FLAG].item() > 0.5
        is_zero2 = emb2[8 + IS_ZERO_FLAG].item() > 0.5

        if is_zero1 or is_zero2:
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
            result = backend.at_add(result, 8 + MAGNITUDE_SIGN, 0.0)
            result = backend.at_add(result, 8 + MAGNITUDE_LOG, -100.0)
        else:
            sign1 = emb1[8 + MAGNITUDE_SIGN].item()
            sign2 = emb2[8 + MAGNITUDE_SIGN].item()
            log1 = emb1[8 + MAGNITUDE_LOG].item()
            log2 = emb2[8 + MAGNITUDE_LOG].item()

            result = backend.at_add(result, 8 + MAGNITUDE_SIGN, sign1 * sign2)
            result = backend.at_add(result, 8 + MAGNITUDE_LOG, log1 + log2)
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 0.0)

        return result

    def divide(self, emb1: Any, emb2: Any) -> Any:
        """
        Divide two physical quantities.

        Dimensions: Subtract exponents
        Magnitude: Divide (subtract log-magnitudes)

        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Subtract dimension exponents
        for i in range(7):
            result = backend.at_add(result, 8 + i, emb1[8 + i] - emb2[8 + i])

        # Update dimensionless flag
        is_dimensionless = all(
            abs(result[8 + i].item()) < 0.5 for i in range(7)
        )
        result = backend.at_add(result, 8 + DIMENSIONLESS_FLAG, 1.0 if is_dimensionless else 0.0)

        # Divide magnitudes (subtract logs)
        is_zero1 = emb1[8 + IS_ZERO_FLAG].item() > 0.5
        is_zero2 = emb2[8 + IS_ZERO_FLAG].item() > 0.5

        if is_zero2:
            # Division by zero - return infinity
            result = backend.at_add(result, 8 + MAGNITUDE_SIGN, emb1[8 + MAGNITUDE_SIGN])
            result = backend.at_add(result, 8 + MAGNITUDE_LOG, float('inf'))
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 0.0)
        elif is_zero1:
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
            result = backend.at_add(result, 8 + MAGNITUDE_SIGN, 0.0)
            result = backend.at_add(result, 8 + MAGNITUDE_LOG, -100.0)
        else:
            sign1 = emb1[8 + MAGNITUDE_SIGN].item()
            sign2 = emb2[8 + MAGNITUDE_SIGN].item()
            log1 = emb1[8 + MAGNITUDE_LOG].item()
            log2 = emb2[8 + MAGNITUDE_LOG].item()

            result = backend.at_add(result, 8 + MAGNITUDE_SIGN, sign1 * sign2)
            result = backend.at_add(result, 8 + MAGNITUDE_LOG, log1 - log2)
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 0.0)

        return result

    def power(self, emb: Any, n: int) -> Any:
        """
        Raise a physical quantity to an integer power.

        Dimensions: Multiply exponents by n
        Magnitude: Raise to power (multiply log by n)

        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Multiply dimension exponents by n
        for i in range(7):
            result = backend.at_add(result, 8 + i, emb[8 + i] * n)

        # Update dimensionless flag
        is_dimensionless = all(
            abs(result[8 + i].item()) < 0.5 for i in range(7)
        )
        result = backend.at_add(result, 8 + DIMENSIONLESS_FLAG, 1.0 if is_dimensionless else 0.0)

        # Power of magnitude
        is_zero = emb[8 + IS_ZERO_FLAG].item() > 0.5

        if is_zero:
            if n > 0:
                result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
                result = backend.at_add(result, 8 + MAGNITUDE_SIGN, 0.0)
                result = backend.at_add(result, 8 + MAGNITUDE_LOG, -100.0)
            else:
                # 0^0 or 0^negative - undefined/infinity
                result = backend.at_add(result, 8 + MAGNITUDE_LOG, float('inf'))
        else:
            sign = emb[8 + MAGNITUDE_SIGN].item()
            log_mag = emb[8 + MAGNITUDE_LOG].item()

            # Sign handling for powers
            if n % 2 == 0:
                result_sign = 1.0  # Even powers are always positive
            else:
                result_sign = sign

            result = backend.at_add(result, 8 + MAGNITUDE_SIGN, result_sign)
            result = backend.at_add(result, 8 + MAGNITUDE_LOG, log_mag * n)
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 0.0)

        return result

    def sqrt(self, emb: Any) -> Any:
        """
        Square root of a physical quantity.

        Dimensions: Halve exponents (must all be even!)
        Magnitude: Square root (halve log)
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Halve dimension exponents (must be even integers)
        exponents = [int(round(emb[8 + i].item())) for i in range(7)]
        for exp in exponents:
            if exp % 2 != 0:
                raise ValueError("Cannot take sqrt of quantity with odd SI dimension exponent")
        for i, exp in enumerate(exponents):
            result = backend.at_add(result, 8 + i, float(exp // 2))

        # Update dimensionless flag
        is_dimensionless = all(
            abs(result[8 + i].item()) < 0.5 for i in range(7)
        )
        result = backend.at_add(result, 8 + DIMENSIONLESS_FLAG, 1.0 if is_dimensionless else 0.0)

        # Square root of magnitude
        is_zero = emb[8 + IS_ZERO_FLAG].item() > 0.5

        if is_zero:
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
            result = backend.at_add(result, 8 + MAGNITUDE_SIGN, 0.0)
            result = backend.at_add(result, 8 + MAGNITUDE_LOG, -100.0)
        else:
            sign = emb[8 + MAGNITUDE_SIGN].item()
            log_mag = emb[8 + MAGNITUDE_LOG].item()

            if sign < 0:
                # Square root of negative - return NaN sentinel
                result = backend.at_add(result, 8 + MAGNITUDE_SIGN, 1.0)
                result = backend.at_add(result, 8 + MAGNITUDE_LOG, float('nan'))
            else:
                result = backend.at_add(result, 8 + MAGNITUDE_SIGN, 1.0)
                result = backend.at_add(result, 8 + MAGNITUDE_LOG, log_mag / 2.0)
                result = backend.at_add(result, 8 + IS_ZERO_FLAG, 0.0)

        return result

    # =========================================================================
    # Type checking
    # =========================================================================

    def can_add(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if two quantities can be added (same dimensions).

        Returns:
            True if dimensions match, False otherwise
        """
        backend = get_backend()
        # Compare dimension exponents
        dims1 = emb1[8:8 + 7]
        dims2 = emb2[8:8 + 7]
        return backend.allclose(dims1, dims2, atol=0.1).item()

    def dimensions_equal(self, emb1: Any, emb2: Any) -> bool:
        """Alias for can_add - check dimensional equality."""
        return self.can_add(emb1, emb2)

    def add(self, emb1: Any, emb2: Any) -> Optional[Any]:
        """
        Add two physical quantities (only if dimensions match).

        Returns None if dimensions don't match (type error).
        """
        if not self.can_add(emb1, emb2):
            return None

        # Decode, add, re-encode
        val1, dims = self.decode(emb1)
        val2, _ = self.decode(emb2)
        return self.encode(val1 + val2, dims)

    def subtract(self, emb1: Any, emb2: Any) -> Optional[Any]:
        """
        Subtract two physical quantities (only if dimensions match).

        Returns None if dimensions don't match (type error).
        """
        if not self.can_add(emb1, emb2):
            return None

        val1, dims = self.decode(emb1)
        val2, _ = self.decode(emb2)
        return self.encode(val1 - val2, dims)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _set_derived_hints(self, emb: Any, dims: Dimensions) -> Any:
        """Set derived unit hint flags based on dimensions."""
        backend = get_backend()
        hints = [
            (FORCE, 0),       # N
            (ENERGY, 1),      # J
            (POWER, 2),       # W
            (PRESSURE, 3),    # Pa
            (VELOCITY, 4),    # m/s
            (ACCELERATION, 5) # m/s²
        ]

        for known_dims, offset in hints:
            if dims == known_dims:
                emb = backend.at_add(emb, 8 + DERIVED_HINTS_OFFSET + offset, 1.0)

        return emb

    def get_derived_unit_name(self, emb: Any) -> Optional[str]:
        """Get the derived SI unit name if recognized."""
        dims = self.decode_dimensions(emb)

        unit_names = {
            DIMENSIONLESS: "dimensionless",
            METER: "m", SECOND: "s", KILOGRAM: "kg",
            AMPERE: "A", KELVIN: "K", MOLE: "mol", CANDELA: "cd",
            VELOCITY: "m/s", ACCELERATION: "m/s²",
            FORCE: "N", ENERGY: "J", POWER: "W",
            PRESSURE: "Pa", FREQUENCY: "Hz",
            CHARGE: "C", VOLTAGE: "V", RESISTANCE: "Ω",
        }

        return unit_names.get(dims)
