"""
Vector Classes and Encoder for FluxEM-Domains

Provides Vector2D and Vector3D with vector operations.
Vectors represent directions and magnitudes in geometric space.

Semantically distinct from Points: Vectors are displacement/direction,
while Points are positions. Operations are mathematically equivalent
but carry different meaning.

Embedding Layout (128 dimensions):
  - dims 0-7:   Domain tag (geom_vector2d or geom_vector3d)
  - dims 8-9:   X component (sign, log_magnitude)
  - dims 10-11: Y component (sign, log_magnitude)
  - dims 12-13: Z component (sign, log_magnitude) - for 3D only
  - dims 14:    Dimensionality (2 or 3)
  - dims 15:    Cached magnitude (log scale)
  - dims 16-17: Cached direction angles
  - dims 72-79: Vector type encoding
  - dims 80-95: Shared semantic features
  - dims 96-127: Cross-domain features
"""

from dataclasses import dataclass
from typing import Any, Union, Tuple
import math
from ...backend import get_backend

from ...core.base import DOMAIN_TAGS, EMBEDDING_DIM, log_encode_value, log_decode_value


@dataclass(frozen=True)
class Vector2D:
    """
    Immutable 2D vector with components.

    Represents a direction and magnitude in 2D space.
    """
    x: float
    y: float

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """Vector addition."""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """Vector subtraction."""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        """Scalar multiplication."""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        """Scalar multiplication (reverse)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        """Scalar division."""
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector2D":
        """Negation (reverse direction)."""
        return Vector2D(-self.x, -self.y)

    def dot(self, other: "Vector2D") -> float:
        """Dot product. a·b = ax*bx + ay*by"""
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector2D") -> float:
        """
        Cross product (z-component of 3D cross product).

        Returns the signed area of the parallelogram.
        """
        return self.x * other.y - self.y * other.x

    def magnitude(self) -> float:
        """Magnitude (length) of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self) -> float:
        """Squared magnitude (avoids sqrt)."""
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vector2D":
        """Return unit vector in same direction."""
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / mag, self.y / mag)

    def perpendicular(self) -> "Vector2D":
        """Return perpendicular vector (90 degrees counter-clockwise)."""
        return Vector2D(-self.y, self.x)

    def rotate(self, angle: float) -> "Vector2D":
        """Rotate vector by angle (radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def angle(self) -> float:
        """Angle from positive x-axis (radians, in [-pi, pi])."""
        return math.atan2(self.y, self.x)

    def angle_to(self, other: "Vector2D") -> float:
        """Angle between this vector and another (radians, in [0, pi])."""
        dot = self.dot(other)
        mags = self.magnitude() * other.magnitude()
        if mags < 1e-10:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / mags))
        return math.acos(cos_angle)

    def signed_angle_to(self, other: "Vector2D") -> float:
        """Signed angle from this vector to another (radians, in [-pi, pi])."""
        return math.atan2(self.cross(other), self.dot(other))

    def project_onto(self, other: "Vector2D") -> "Vector2D":
        """Project this vector onto another."""
        dot_product = self.dot(other)
        other_mag_sq = other.magnitude_squared()
        if other_mag_sq < 1e-20:
            return Vector2D(0.0, 0.0)
        scale = dot_product / other_mag_sq
        return other * scale

    def reject_from(self, other: "Vector2D") -> "Vector2D":
        """Component of this vector perpendicular to another."""
        return self - self.project_onto(other)

    def reflect(self, normal: "Vector2D") -> "Vector2D":
        """Reflect this vector about a normal."""
        n = normal.normalized()
        return self - 2 * self.dot(n) * n

    def is_parallel(self, other: "Vector2D", tolerance: float = 1e-9) -> bool:
        """Check if vectors are parallel."""
        return abs(self.cross(other)) < tolerance

    def is_perpendicular(self, other: "Vector2D", tolerance: float = 1e-9) -> bool:
        """Check if vectors are perpendicular."""
        return abs(self.dot(other)) < tolerance

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"


@dataclass(frozen=True)
class Vector3D:
    """
    Immutable 3D vector with components.

    Represents a direction and magnitude in 3D space.
    """
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3D") -> "Vector3D":
        """Vector addition."""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        """Vector subtraction."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3D":
        """Scalar multiplication."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3D":
        """Scalar multiplication (reverse)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3D":
        """Scalar division."""
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3D":
        """Negation (reverse direction)."""
        return Vector3D(-self.x, -self.y, -self.z)

    def dot(self, other: "Vector3D") -> float:
        """Dot product. a·b = ax*bx + ay*by + az*bz"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3D") -> "Vector3D":
        """
        Cross product.

        Returns vector perpendicular to both inputs.
        |a × b| = |a||b|sin(θ)
        """
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def triple_product(self, b: "Vector3D", c: "Vector3D") -> float:
        """
        Scalar triple product: self · (b × c).

        Returns signed volume of parallelepiped.
        """
        return self.dot(b.cross(c))

    def magnitude(self) -> float:
        """Magnitude (length) of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def magnitude_squared(self) -> float:
        """Squared magnitude (avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> "Vector3D":
        """Return unit vector in same direction."""
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)

    def angle_to(self, other: "Vector3D") -> float:
        """Angle between this vector and another (radians)."""
        dot = self.dot(other)
        mags = self.magnitude() * other.magnitude()
        if mags < 1e-10:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / mags))
        return math.acos(cos_angle)

    def project_onto(self, other: "Vector3D") -> "Vector3D":
        """Project this vector onto another."""
        dot_product = self.dot(other)
        other_mag_sq = other.magnitude_squared()
        if other_mag_sq < 1e-20:
            return Vector3D(0.0, 0.0, 0.0)
        scale = dot_product / other_mag_sq
        return other * scale

    def reject_from(self, other: "Vector3D") -> "Vector3D":
        """Component of this vector perpendicular to another."""
        return self - self.project_onto(other)

    def reflect(self, normal: "Vector3D") -> "Vector3D":
        """Reflect this vector about a normal."""
        n = normal.normalized()
        return self - 2 * self.dot(n) * n

    def is_parallel(self, other: "Vector3D", tolerance: float = 1e-9) -> bool:
        """Check if vectors are parallel."""
        cross = self.cross(other)
        return cross.magnitude() < tolerance

    def is_perpendicular(self, other: "Vector3D", tolerance: float = 1e-9) -> bool:
        """Check if vectors are perpendicular."""
        return abs(self.dot(other)) < tolerance

    def rotate_around_axis(self, axis: "Vector3D", angle: float) -> "Vector3D":
        """
        Rotate this vector around an axis by angle (radians).

        Uses Rodrigues' rotation formula for the given angle.
        """
        k = axis.normalized()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1 - cos(θ))
        term1 = self * cos_a
        term2 = k.cross(self) * sin_a
        term3 = k * (k.dot(self) * (1 - cos_a))

        return term1 + term2 + term3

    def to_spherical(self) -> Tuple[float, float, float]:
        """Convert to spherical coordinates (r, theta, phi)."""
        r = self.magnitude()
        if r < 1e-10:
            return (0.0, 0.0, 0.0)
        theta = math.acos(max(-1.0, min(1.0, self.z / r)))  # polar angle
        phi = math.atan2(self.y, self.x)  # azimuthal angle
        return (r, theta, phi)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return f"Vector3D({self.x}, {self.y}, {self.z})"


class VectorEncoder:
    """
    Encoder for 2D and 3D vectors.

    Encodes vectors using log-magnitude representation for components.
    """

    # Embedding layout - absolute positions
    X_SIGN_POS = 8
    X_LOG_POS = 9
    Y_SIGN_POS = 10
    Y_LOG_POS = 11
    Z_SIGN_POS = 12
    Z_LOG_POS = 13
    DIMENSION_POS = 14
    MAGNITUDE_POS = 15
    ANGLE_POS = 16  # Direction angle (2D) or azimuthal (3D)
    POLAR_POS = 17  # Polar angle (3D only)
    TYPE_START = 72

    def __init__(self):
        """Initialize the vector encoder."""
        pass

    def encode(self, vector: Union[Vector2D, Vector3D]) -> Any:
        """
        Encode a vector as a 128-dimensional embedding.
        """
        backend = get_backend()
        embedding = backend.zeros((EMBEDDING_DIM,))

        is_3d = isinstance(vector, Vector3D)

        # Set domain tag
        tag_key = "geom_vector3d" if is_3d else "geom_vector2d"
        if tag_key in DOMAIN_TAGS:
            embedding = backend.at_add(embedding, slice(0, 8), DOMAIN_TAGS[tag_key])

        # Encode X component
        x_sign, x_log = log_encode_value(vector.x)
        embedding = backend.at_add(embedding, self.X_SIGN_POS, x_sign)
        embedding = backend.at_add(embedding, self.X_LOG_POS, x_log)

        # Encode Y component
        y_sign, y_log = log_encode_value(vector.y)
        embedding = backend.at_add(embedding, self.Y_SIGN_POS, y_sign)
        embedding = backend.at_add(embedding, self.Y_LOG_POS, y_log)

        # Encode Z component (3D only)
        if is_3d:
            z_sign, z_log = log_encode_value(vector.z)
            embedding = backend.at_add(embedding, self.Z_SIGN_POS, z_sign)
            embedding = backend.at_add(embedding, self.Z_LOG_POS, z_log)

        # Store dimensionality
        dim = 3.0 if is_3d else 2.0
        embedding = backend.at_add(embedding, self.DIMENSION_POS, dim)

        # Store magnitude
        mag = vector.magnitude()
        if mag > 0:
            _, log_mag = log_encode_value(mag)
            embedding = backend.at_add(embedding, self.MAGNITUDE_POS, log_mag)

        # Store direction angle
        if is_3d:
            r, theta, phi = vector.to_spherical()
            embedding = backend.at_add(embedding, self.ANGLE_POS, phi)
            embedding = backend.at_add(embedding, self.POLAR_POS, theta)
        else:
            embedding = backend.at_add(embedding, self.ANGLE_POS, vector.angle())

        return embedding

    def decode(self, embedding: Any) -> Union[Vector2D, Vector3D]:
        """
        Decode an embedding back to a Vector.
        """
        dim = int(float(embedding[self.DIMENSION_POS]) + 0.5)
        is_3d = dim == 3

        # Decode X
        x_sign = float(embedding[self.X_SIGN_POS])
        x_log = float(embedding[self.X_LOG_POS])
        x = log_decode_value(x_sign, x_log)

        # Decode Y
        y_sign = float(embedding[self.Y_SIGN_POS])
        y_log = float(embedding[self.Y_LOG_POS])
        y = log_decode_value(y_sign, y_log)

        if is_3d:
            # Decode Z
            z_sign = float(embedding[self.Z_SIGN_POS])
            z_log = float(embedding[self.Z_LOG_POS])
            z = log_decode_value(z_sign, z_log)
            return Vector3D(x, y, z)
        else:
            return Vector2D(x, y)

    # Vector operations

    def add(self, emb1: Any, emb2: Any) -> Any:
        """Add two vectors."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)
        return self.encode(v1 + v2)

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """Subtract two vectors."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)
        return self.encode(v1 - v2)

    def scale(self, embedding: Any, scalar: float) -> Any:
        """Scale a vector by a scalar."""
        v = self.decode(embedding)
        return self.encode(v * scalar)

    def negate(self, embedding: Any) -> Any:
        """Negate a vector."""
        v = self.decode(embedding)
        return self.encode(-v)

    def dot_product(self, emb1: Any, emb2: Any) -> float:
        """Compute dot product."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)
        return v1.dot(v2)

    def cross_product(self, emb1: Any, emb2: Any) -> Any:
        """
        Compute cross product.

        For 2D: returns scalar (z-component)
        For 3D: returns Vector3D embedding
        """
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)

        if isinstance(v1, Vector3D) and isinstance(v2, Vector3D):
            result = v1.cross(v2)
            return self.encode(result)
        elif isinstance(v1, Vector2D) and isinstance(v2, Vector2D):
            # Return the scalar as a 1D vector in z
            scalar = v1.cross(v2)
            return self.encode(Vector3D(0, 0, scalar))
        else:
            raise ValueError("Cross product requires matching dimensions")

    def magnitude(self, embedding: Any) -> float:
        """Get magnitude of vector."""
        v = self.decode(embedding)
        return v.magnitude()

    def normalize(self, embedding: Any) -> Any:
        """Normalize to unit vector (preserves direction)."""
        v = self.decode(embedding)
        return self.encode(v.normalized())

    def angle_between(self, emb1: Any, emb2: Any) -> float:
        """Angle between two vectors (radians) from dot product."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)
        return v1.angle_to(v2)

    def project(self, emb1: Any, emb2: Any) -> Any:
        """Project first vector onto second."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)
        return self.encode(v1.project_onto(v2))

    def reflect(self, embedding: Any, normal: Any) -> Any:
        """Reflect vector about normal."""
        v = self.decode(embedding)
        n = self.decode(normal)

        if isinstance(v, Vector3D) and isinstance(n, Vector3D):
            return self.encode(v.reflect(n))
        elif isinstance(v, Vector2D) and isinstance(n, Vector2D):
            return self.encode(v.reflect(n))
        else:
            raise ValueError("Reflection requires matching dimensions")

    def is_parallel(self, emb1: Any, emb2: Any, tolerance: float = 1e-9) -> bool:
        """Check if two vectors are parallel."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)
        return v1.is_parallel(v2, tolerance)

    def is_perpendicular(self, emb1: Any, emb2: Any, tolerance: float = 1e-9) -> bool:
        """Check if two vectors are perpendicular."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)
        return v1.is_perpendicular(v2, tolerance)

    def equals(self, emb1: Any, emb2: Any, tolerance: float = 1e-9) -> bool:
        """Check if two vector embeddings represent the same vector."""
        v1 = self.decode(emb1)
        v2 = self.decode(emb2)
        diff = v1 - v2
        return diff.magnitude() < tolerance


# Standard basis vectors
UNIT_X_2D = Vector2D(1.0, 0.0)
UNIT_Y_2D = Vector2D(0.0, 1.0)
ZERO_2D = Vector2D(0.0, 0.0)

UNIT_X_3D = Vector3D(1.0, 0.0, 0.0)
UNIT_Y_3D = Vector3D(0.0, 1.0, 0.0)
UNIT_Z_3D = Vector3D(0.0, 0.0, 1.0)
ZERO_3D = Vector3D(0.0, 0.0, 0.0)


def from_spherical(r: float, theta: float, phi: float) -> Vector3D:
    """Create a 3D vector from spherical coordinates."""
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return Vector3D(x, y, z)


def from_polar(r: float, theta: float) -> Vector2D:
    """Create a 2D vector from polar coordinates."""
    return Vector2D(r * math.cos(theta), r * math.sin(theta))
