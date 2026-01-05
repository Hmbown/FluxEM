"""
Vector Encoder.

Embeds vectors with component-wise log-magnitude representation.
Supports vectors up to 16 dimensions.

Addition is defined in embedding space when dimensionality matches.
Dot product requires decode-operate-encode.
"""

import math
from typing import Any, List, Tuple, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


# Maximum vector dimension we can encode
MAX_VECTOR_DIM = 16

# Embedding layout within domain-specific region (dims 8-71):
# dims 0-1:   Dimension (n) encoded as (1.0, n/16.0)
# dims 2-3:   Norm info (sign=1, log|v|)
# dims 4-5:   Reserved for flags
# dims 6-37:  Components (16 pairs of sign, log|component|)
# dims 38-63: Reserved

DIM_OFFSET = 0
NORM_OFFSET = 2
FLAGS_OFFSET = 4
COMPONENTS_OFFSET = 6


class VectorEncoder:
    """
    Encoder for real vectors.

    Uses log-magnitude representation for each component.
    Supports vectors up to 16 dimensions.

    Component-wise operations (addition, scalar multiplication) use component arithmetic.
    """

    domain_tag = DOMAIN_TAGS["math_vector"]
    domain_name = "math_vector"

    def encode(self, v: Union[List[float], Tuple[float, ...], Any]) -> Any:
        """
        Encode a vector.

        Args:
            v: Vector as list, tuple, or array (max 16 dimensions)

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if hasattr(v, "tolist"):
            v = v.tolist()
        v = list(v)

        n = len(v)
        if n > MAX_VECTOR_DIM:
            raise ValueError(f"Vector dimension {n} exceeds maximum {MAX_VECTOR_DIM}")
        if n == 0:
            raise ValueError("Cannot encode empty vector")

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Dimension encoding
        emb = backend.at_add(emb, 8 + DIM_OFFSET, 1.0)
        emb = backend.at_add(emb, 8 + DIM_OFFSET + 1, n / MAX_VECTOR_DIM)

        # Compute and encode norm
        norm_sq = sum(x * x for x in v)
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 0.0

        if norm < EPSILON:
            emb = backend.at_add(emb, 8 + NORM_OFFSET, 0.0)
            emb = backend.at_add(emb, 8 + NORM_OFFSET + 1, -100.0)  # Zero sentinel
        else:
            emb = backend.at_add(emb, 8 + NORM_OFFSET, 1.0)
            emb = backend.at_add(emb, 8 + NORM_OFFSET + 1, math.log(norm))

        # Encode each component
        for i, val in enumerate(v):
            sign, log_mag = log_encode_value(val)
            emb = backend.at_add(emb, 8 + COMPONENTS_OFFSET + 2*i, sign)
            emb = backend.at_add(emb, 8 + COMPONENTS_OFFSET + 2*i + 1, log_mag)

        # Zero out remaining component slots (already zero from create_embedding)

        return emb

    def decode(self, emb: Any) -> List[float]:
        """
        Decode embedding to vector.

        Returns:
            List of floats representing the vector
        """
        # Get dimension
        n = int(round(emb[8 + DIM_OFFSET + 1].item() * MAX_VECTOR_DIM))
        if n < 1:
            n = 1
        if n > MAX_VECTOR_DIM:
            n = MAX_VECTOR_DIM

        # Decode each component
        result = []
        for i in range(n):
            sign = emb[8 + COMPONENTS_OFFSET + 2*i].item()
            log_mag = emb[8 + COMPONENTS_OFFSET + 2*i + 1].item()
            result.append(log_decode_value(sign, log_mag))

        return result

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid vector."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    def get_dimension(self, emb: Any) -> int:
        """Get the dimension of the encoded vector."""
        return int(round(emb[8 + DIM_OFFSET + 1].item() * MAX_VECTOR_DIM))

    def get_norm(self, emb: Any) -> float:
        """Get the Euclidean norm of the vector."""
        sign = emb[8 + NORM_OFFSET].item()
        log_mag = emb[8 + NORM_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    # =========================================================================
    # Operations
    # =========================================================================

    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two vectors (component-wise).

        Requires decode-operate-encode due to log representation.
        """
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)

        n1, n2 = len(v1), len(v2)
        if n1 != n2:
            raise ValueError(f"Cannot add vectors of different dimensions: {n1} vs {n2}")

        result = [a + b for a, b in zip(v1, v2)]
        return self.encode(result)

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """Subtract two vectors (component-wise)."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)

        n1, n2 = len(v1), len(v2)
        if n1 != n2:
            raise ValueError(f"Cannot subtract vectors of different dimensions: {n1} vs {n2}")

        result = [a - b for a, b in zip(v1, v2)]
        return self.encode(result)

    def scale(self, emb: Any, scalar: float) -> Any:
        """
        Multiply vector by scalar.

        Computed in log-space: add log(|scalar|) to each log-magnitude.
        """
        backend = get_backend()
        if abs(scalar) < EPSILON:
            # Result is zero vector
            v = self.decode(emb)
            return self.encode([0.0] * len(v))

        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Copy dimension
        result = backend.at_add(result, 8 + DIM_OFFSET, emb[8 + DIM_OFFSET])
        result = backend.at_add(result, 8 + DIM_OFFSET + 1, emb[8 + DIM_OFFSET + 1])

        n = self.get_dimension(emb)
        scalar_sign = 1.0 if scalar > 0 else -1.0
        log_scalar = math.log(abs(scalar))

        # Scale each component
        for i in range(n):
            sign = emb[8 + COMPONENTS_OFFSET + 2*i].item()
            log_mag = emb[8 + COMPONENTS_OFFSET + 2*i + 1].item()

            if abs(sign) < 0.5:
                # Zero component stays zero
                result = backend.at_add(result, 8 + COMPONENTS_OFFSET + 2*i, 0.0)
                result = backend.at_add(result, 8 + COMPONENTS_OFFSET + 2*i + 1, -100.0)
            else:
                new_sign = sign * scalar_sign
                new_log_mag = log_mag + log_scalar
                result = backend.at_add(result, 8 + COMPONENTS_OFFSET + 2*i, new_sign)
                result = backend.at_add(result, 8 + COMPONENTS_OFFSET + 2*i + 1, new_log_mag)

        # Update norm (scale by |scalar|)
        old_norm_sign = emb[8 + NORM_OFFSET].item()
        old_log_norm = emb[8 + NORM_OFFSET + 1].item()
        if abs(old_norm_sign) < 0.5:
            result = backend.at_add(result, 8 + NORM_OFFSET, 0.0)
            result = backend.at_add(result, 8 + NORM_OFFSET + 1, -100.0)
        else:
            result = backend.at_add(result, 8 + NORM_OFFSET, 1.0)
            result = backend.at_add(result, 8 + NORM_OFFSET + 1, old_log_norm + log_scalar)

        return result

    def negate(self, emb: Any) -> Any:
        """Negate the vector (multiply by -1)."""
        return self.scale(emb, -1.0)

    def dot(self, emb1: Any, emb2: Any) -> float:
        """
        Compute dot product of two vectors.

        Requires decode (not a homomorphism in log-space).
        """
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)

        n1, n2 = len(v1), len(v2)
        if n1 != n2:
            raise ValueError(f"Cannot dot vectors of different dimensions: {n1} vs {n2}")

        return sum(a * b for a, b in zip(v1, v2))

    def cross(self, emb1: Any, emb2: Any) -> Any:
        """
        Compute cross product of two 3D vectors.

        Only valid for 3-dimensional vectors.
        """
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)

        if len(v1) != 3 or len(v2) != 3:
            raise ValueError("Cross product only defined for 3D vectors")

        result = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]
        return self.encode(result)

    def normalize(self, emb: Any) -> Any:
        """
        Return unit vector in same direction.

        Returns zero vector if input is zero.
        """
        norm = self.get_norm(emb)
        if norm < EPSILON:
            return emb  # Keep zero vector as-is
        return self.scale(emb, 1.0 / norm)

    def is_zero(self, emb: Any) -> bool:
        """Check if the vector is zero (all components zero)."""
        return self.get_norm(emb) < EPSILON

    def is_unit(self, emb: Any, tol: float = 1e-6) -> bool:
        """Check if the vector is a unit vector."""
        norm = self.get_norm(emb)
        return abs(norm - 1.0) < tol

    def angle_between(self, emb1: Any, emb2: Any) -> float:
        """
        Compute angle between two vectors in radians.

        Returns 0 if either vector is zero.
        """
        n1 = self.get_norm(emb1)
        n2 = self.get_norm(emb2)

        if n1 < EPSILON or n2 < EPSILON:
            return 0.0

        dot_val = self.dot(emb1, emb2)
        cos_angle = dot_val / (n1 * n2)

        # Clamp to [-1, 1] for numerical stability
        cos_angle = max(-1.0, min(1.0, cos_angle))

        return math.acos(cos_angle)

    def project(self, emb1: Any, emb2: Any) -> Any:
        """
        Project emb1 onto emb2.

        proj_v2(v1) = (v1 · v2 / |v2|^2) * v2
        """
        n2_sq = self.get_norm(emb2) ** 2
        if n2_sq < EPSILON:
            # Cannot project onto zero vector
            v1 = self.decode(emb1)
            return self.encode([0.0] * len(v1))

        dot_val = self.dot(emb1, emb2)
        scalar = dot_val / n2_sq
        return self.scale(emb2, scalar)

    def component(self, emb: Any, index: int) -> float:
        """Get a specific component of the vector."""
        n = self.get_dimension(emb)
        if index < 0 or index >= n:
            raise IndexError(f"Component index {index} out of range [0, {n})")

        sign = emb[8 + COMPONENTS_OFFSET + 2*index].item()
        log_mag = emb[8 + COMPONENTS_OFFSET + 2*index + 1].item()
        return log_decode_value(sign, log_mag)
