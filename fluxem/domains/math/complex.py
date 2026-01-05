"""
Complex Number Encoder.

Embeds complex numbers with log-magnitude and phase representation.
Multiplication becomes: add log-magnitudes, add phases (rotation).

Multiplication is implemented in embedding space.
Addition requires decode-operate-encode (not a group homomorphism in log-polar).
"""

import cmath
import math
from typing import Any, Tuple, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
)


# Embedding layout:
# dims 0-1:   Log-magnitude (sign=1 for r>0, log|z|)
# dims 2-3:   Phase (cos θ, sin θ) - unit circle representation
# dims 4-5:   Real part (sign, log|Re(z)|)
# dims 6-7:   Imaginary part (sign, log|Im(z)|)
# dims 8:     Is zero flag
# dims 9:     Is real flag (Im(z) = 0)
# dims 10:    Is pure imaginary flag (Re(z) = 0)
# dims 11-15: Reserved

LOG_MAG_OFFSET = 0
PHASE_OFFSET = 2
REAL_OFFSET = 4
IMAG_OFFSET = 6
IS_ZERO_FLAG = 8
IS_REAL_FLAG = 9
IS_PURE_IMAG_FLAG = 10


class ComplexEncoder:
    """
    Encoder for complex numbers.

    Uses log-polar representation for efficient multiplication:
    - z = r * e^(iθ) where r = |z|, θ = arg(z)
    - Multiplication: z1 * z2 = r1*r2 * e^(i(θ1+θ2))
        -> Add log-magnitudes, add phases

    Multiplication is implemented in embedding space.
    """

    domain_tag = DOMAIN_TAGS["math_complex"]
    domain_name = "math_complex"

    def encode(self, z: Union[complex, float, Tuple[float, float]]) -> Any:
        """
        Encode a complex number.

        Args:
            z: Complex number, float (real), or (real, imag) tuple

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(z, tuple):
            z = complex(z[0], z[1])
        elif isinstance(z, (int, float)):
            z = complex(z, 0)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        r = abs(z)
        theta = cmath.phase(z)

        # Handle zero
        if r < EPSILON:
            emb = backend.at_add(emb, 8 + IS_ZERO_FLAG, 1.0)
            emb = backend.at_add(emb, 8 + LOG_MAG_OFFSET, 0.0)
            emb = backend.at_add(emb, 8 + LOG_MAG_OFFSET + 1, -100.0)  # Zero sentinel
            emb = backend.at_add(emb, 8 + PHASE_OFFSET, 1.0)  # cos(0)
            emb = backend.at_add(emb, 8 + PHASE_OFFSET + 1, 0.0)  # sin(0)
        else:
            # Log-magnitude
            emb = backend.at_add(emb, 8 + LOG_MAG_OFFSET, 1.0)  # Magnitude is positive
            emb = backend.at_add(emb, 8 + LOG_MAG_OFFSET + 1, math.log(r))

            # Phase on unit circle representation
            emb = backend.at_add(emb, 8 + PHASE_OFFSET, math.cos(theta))
            emb = backend.at_add(emb, 8 + PHASE_OFFSET + 1, math.sin(theta))

        # Real and imaginary parts (for reconstruction stability)
        re = z.real
        im = z.imag

        if abs(re) < EPSILON:
            emb = backend.at_add(emb, 8 + REAL_OFFSET, 0.0)
            emb = backend.at_add(emb, 8 + REAL_OFFSET + 1, -100.0)
        else:
            emb = backend.at_add(emb, 8 + REAL_OFFSET, 1.0 if re > 0 else -1.0)
            emb = backend.at_add(emb, 8 + REAL_OFFSET + 1, math.log(abs(re)))

        if abs(im) < EPSILON:
            emb = backend.at_add(emb, 8 + IMAG_OFFSET, 0.0)
            emb = backend.at_add(emb, 8 + IMAG_OFFSET + 1, -100.0)
        else:
            emb = backend.at_add(emb, 8 + IMAG_OFFSET, 1.0 if im > 0 else -1.0)
            emb = backend.at_add(emb, 8 + IMAG_OFFSET + 1, math.log(abs(im)))

        # Flags
        emb = backend.at_add(emb, 8 + IS_REAL_FLAG, 1.0 if abs(im) < EPSILON else 0.0)
        emb = backend.at_add(emb, 8 + IS_PURE_IMAG_FLAG, 
            1.0 if abs(re) < EPSILON and abs(im) >= EPSILON else 0.0
        )

        return emb

    def decode(self, emb: Any) -> complex:
        """
        Decode embedding to complex number.

        Uses polar form for primary reconstruction.
        """
        is_zero = emb[8 + IS_ZERO_FLAG].item() > 0.5
        if is_zero:
            return complex(0, 0)

        # Reconstruct from polar form
        log_r = emb[8 + LOG_MAG_OFFSET + 1].item()
        r = math.exp(log_r)

        cos_theta = emb[8 + PHASE_OFFSET].item()
        sin_theta = emb[8 + PHASE_OFFSET + 1].item()
        theta = math.atan2(sin_theta, cos_theta)

        return r * cmath.exp(1j * theta)

    def decode_polar(self, emb: Any) -> Tuple[float, float]:
        """Decode to (magnitude, phase)."""
        is_zero = emb[8 + IS_ZERO_FLAG].item() > 0.5
        if is_zero:
            return (0.0, 0.0)

        log_r = emb[8 + LOG_MAG_OFFSET + 1].item()
        r = math.exp(log_r)

        cos_theta = emb[8 + PHASE_OFFSET].item()
        sin_theta = emb[8 + PHASE_OFFSET + 1].item()
        theta = math.atan2(sin_theta, cos_theta)

        return (r, theta)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid complex number."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Operations
    # =========================================================================

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two complex numbers.

        z1 * z2: magnitudes multiply (logs add), phases add.
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        is_zero1 = emb1[8 + IS_ZERO_FLAG].item() > 0.5
        is_zero2 = emb2[8 + IS_ZERO_FLAG].item() > 0.5

        if is_zero1 or is_zero2:
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
            result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, -100.0)
            result = backend.at_add(result, 8 + PHASE_OFFSET, 1.0)
            return result

        # Add log-magnitudes
        log_r1 = emb1[8 + LOG_MAG_OFFSET + 1].item()
        log_r2 = emb2[8 + LOG_MAG_OFFSET + 1].item()
        result = backend.at_add(result, 8 + LOG_MAG_OFFSET, 1.0)
        result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, log_r1 + log_r2)

        # Add phases (rotate): (cos θ1, sin θ1) * (cos θ2, sin θ2)
        cos1, sin1 = emb1[8 + PHASE_OFFSET].item(), emb1[8 + PHASE_OFFSET + 1].item()
        cos2, sin2 = emb2[8 + PHASE_OFFSET].item(), emb2[8 + PHASE_OFFSET + 1].item()

        # Rotation formula
        cos_result = cos1 * cos2 - sin1 * sin2
        sin_result = sin1 * cos2 + cos1 * sin2

        result = backend.at_add(result, 8 + PHASE_OFFSET, cos_result)
        result = backend.at_add(result, 8 + PHASE_OFFSET + 1, sin_result)

        # Update real/imag (decode and re-encode)
        z = self.decode(result)
        if abs(z.real) >= EPSILON:
            result = backend.at_add(result, 8 + REAL_OFFSET, 1.0 if z.real > 0 else -1.0)
            result = backend.at_add(result, 8 + REAL_OFFSET + 1, math.log(abs(z.real)))
        if abs(z.imag) >= EPSILON:
            result = backend.at_add(result, 8 + IMAG_OFFSET, 1.0 if z.imag > 0 else -1.0)
            result = backend.at_add(result, 8 + IMAG_OFFSET + 1, math.log(abs(z.imag)))

        return result

    def divide(self, emb1: Any, emb2: Any) -> Any:
        """
        Divide two complex numbers.

        z1 / z2: magnitudes divide (logs subtract), phases subtract.
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        is_zero1 = emb1[8 + IS_ZERO_FLAG].item() > 0.5
        is_zero2 = emb2[8 + IS_ZERO_FLAG].item() > 0.5

        if is_zero2:
            # Division by zero -> infinity
            result = backend.at_add(result, 8 + LOG_MAG_OFFSET, 1.0)
            result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, 100.0)  # Infinity
            return result

        if is_zero1:
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
            result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, -100.0)
            result = backend.at_add(result, 8 + PHASE_OFFSET, 1.0)
            return result

        # Subtract log-magnitudes
        log_r1 = emb1[8 + LOG_MAG_OFFSET + 1].item()
        log_r2 = emb2[8 + LOG_MAG_OFFSET + 1].item()
        result = backend.at_add(result, 8 + LOG_MAG_OFFSET, 1.0)
        result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, log_r1 - log_r2)

        # Subtract phases (counter-rotate)
        cos1, sin1 = emb1[8 + PHASE_OFFSET].item(), emb1[8 + PHASE_OFFSET + 1].item()
        cos2, sin2 = emb2[8 + PHASE_OFFSET].item(), emb2[8 + PHASE_OFFSET + 1].item()

        # Division = multiply by conjugate normalized: (cos -θ2, sin -θ2) = (cos2, -sin2)
        cos_result = cos1 * cos2 + sin1 * sin2
        sin_result = sin1 * cos2 - cos1 * sin2

        result = backend.at_add(result, 8 + PHASE_OFFSET, cos_result)
        result = backend.at_add(result, 8 + PHASE_OFFSET + 1, sin_result)

        return result

    def conjugate(self, emb: Any) -> Any:
        """
        Complex conjugate.

        Negates the phase (imaginary part).
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Copy magnitude
        result = backend.at_add(result, 8 + LOG_MAG_OFFSET, emb[8 + LOG_MAG_OFFSET])
        result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, emb[8 + LOG_MAG_OFFSET + 1])

        # Negate phase (negate sin, keep cos)
        result = backend.at_add(result, 8 + PHASE_OFFSET, emb[8 + PHASE_OFFSET])
        result = backend.at_add(result, 8 + PHASE_OFFSET + 1, -emb[8 + PHASE_OFFSET + 1])

        # Copy real, negate imaginary
        result = backend.at_add(result, 8 + REAL_OFFSET, emb[8 + REAL_OFFSET])
        result = backend.at_add(result, 8 + REAL_OFFSET + 1, emb[8 + REAL_OFFSET + 1])
        result = backend.at_add(result, 8 + IMAG_OFFSET, -emb[8 + IMAG_OFFSET])
        result = backend.at_add(result, 8 + IMAG_OFFSET + 1, emb[8 + IMAG_OFFSET + 1])

        # Copy flags
        result = backend.at_add(result, 8 + IS_ZERO_FLAG, emb[8 + IS_ZERO_FLAG])
        result = backend.at_add(result, 8 + IS_REAL_FLAG, emb[8 + IS_REAL_FLAG])

        return result

    def magnitude(self, emb: Any) -> float:
        """Extract the magnitude |z|."""
        if emb[8 + IS_ZERO_FLAG].item() > 0.5:
            return 0.0
        log_r = emb[8 + LOG_MAG_OFFSET + 1].item()
        return math.exp(log_r)

    def phase(self, emb: Any) -> float:
        """Extract the phase arg(z)."""
        cos_theta = emb[8 + PHASE_OFFSET].item()
        sin_theta = emb[8 + PHASE_OFFSET + 1].item()
        return math.atan2(sin_theta, cos_theta)

    def power(self, emb: Any, n: int) -> Any:
        """
        Raise to integer power.

        z^n: magnitude^n, phase*n
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        if emb[8 + IS_ZERO_FLAG].item() > 0.5:
            if n > 0:
                result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
                result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, -100.0)
            else:
                result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, 100.0)  # Infinity
            return result

        # Scale log-magnitude
        log_r = emb[8 + LOG_MAG_OFFSET + 1].item()
        result = backend.at_add(result, 8 + LOG_MAG_OFFSET, 1.0)
        result = backend.at_add(result, 8 + LOG_MAG_OFFSET + 1, log_r * n)

        # Scale phase
        cos_theta = emb[8 + PHASE_OFFSET].item()
        sin_theta = emb[8 + PHASE_OFFSET + 1].item()
        theta = math.atan2(sin_theta, cos_theta)
        new_theta = theta * n

        result = backend.at_add(result, 8 + PHASE_OFFSET, math.cos(new_theta))
        result = backend.at_add(result, 8 + PHASE_OFFSET + 1, math.sin(new_theta))

        return result

    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two complex numbers.

        Note: This requires decode-operate-encode (not a homomorphism in log-polar).
        """
        z1 = self.decode(emb1)
        z2 = self.decode(emb2)
        return self.encode(z1 + z2)

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """Subtract two complex numbers."""
        z1 = self.decode(emb1)
        z2 = self.decode(emb2)
        return self.encode(z1 - z2)

    def is_real(self, emb: Any) -> bool:
        """Check if the number is real."""
        return emb[8 + IS_REAL_FLAG].item() > 0.5

    def is_pure_imaginary(self, emb: Any) -> bool:
        """Check if the number is pure imaginary."""
        return emb[8 + IS_PURE_IMAG_FLAG].item() > 0.5
