"""
Point Classes and Encoder for FluxEM-Domains

Provides Point2D and Point3D with coordinate representation.
Points are immutable and support standard geometric operations.

Embedding Layout (128 dimensions):
  - dims 0-7:   Domain tag (geom_point2d or geom_point3d)
  - dims 8-9:   X coordinate (sign, log_magnitude)
  - dims 10-11: Y coordinate (sign, log_magnitude)
  - dims 12-13: Z coordinate (sign, log_magnitude) - for 3D only
  - dims 14-15: Reserved for precision/scale
  - dims 16-71: Additional point features
  - dims 72-79: Point type encoding
  - dims 80-95: Shared semantic features
  - dims 96-127: Cross-domain features
"""

from dataclasses import dataclass
from typing import Any, Union, Tuple, Optional
import math
from ...backend import get_backend

from ...core.base import DOMAIN_TAGS, EMBEDDING_DIM, log_encode_value, log_decode_value


@dataclass(frozen=True)
class Point2D:
    """
    Immutable 2D point with coordinates.

    Supports standard geometric operations like distance, midpoint, etc.
    """
    x: float
    y: float

    def __add__(self, other: "Point2D") -> "Point2D":
        """Vector addition (treating points as position vectors)."""
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point2D") -> "Point2D":
        """Vector subtraction."""
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point2D":
        """Scalar multiplication."""
        return Point2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Point2D":
        """Scalar multiplication (reverse)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Point2D":
        """Scalar division."""
        return Point2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Point2D":
        """Negation."""
        return Point2D(-self.x, -self.y)

    def distance_to(self, other: "Point2D") -> float:
        """Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def manhattan_distance_to(self, other: "Point2D") -> float:
        """Manhattan distance to another point."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def midpoint(self, other: "Point2D") -> "Point2D":
        """Midpoint between this point and another."""
        return Point2D((self.x + other.x) / 2, (self.y + other.y) / 2)

    def dot(self, other: "Point2D") -> float:
        """Dot product (treating points as vectors)."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Point2D") -> float:
        """Cross product z-component (treating points as vectors in xy-plane)."""
        return self.x * other.y - self.y * other.x

    def magnitude(self) -> float:
        """Magnitude (distance from origin)."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> "Point2D":
        """Return unit vector in same direction."""
        mag = self.magnitude()
        if mag < 1e-10:
            return Point2D(0.0, 0.0)
        return Point2D(self.x / mag, self.y / mag)

    def rotate(self, angle: float) -> "Point2D":
        """Rotate point around origin by angle (radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Point2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def angle(self) -> float:
        """Angle from positive x-axis (radians)."""
        return math.atan2(self.y, self.x)

    def angle_to(self, other: "Point2D") -> float:
        """Angle between this vector and another (radians)."""
        dot = self.dot(other)
        mags = self.magnitude() * other.magnitude()
        if mags < 1e-10:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / mags))
        return math.acos(cos_angle)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Point2D({self.x}, {self.y})"


@dataclass(frozen=True)
class Point3D:
    """
    Immutable 3D point with coordinates.

    Supports standard geometric operations in 3D space.
    """
    x: float
    y: float
    z: float

    def __add__(self, other: "Point3D") -> "Point3D":
        """Vector addition."""
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Point3D") -> "Point3D":
        """Vector subtraction."""
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Point3D":
        """Scalar multiplication."""
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Point3D":
        """Scalar multiplication (reverse)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Point3D":
        """Scalar division."""
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Point3D":
        """Negation."""
        return Point3D(-self.x, -self.y, -self.z)

    def distance_to(self, other: "Point3D") -> float:
        """Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def manhattan_distance_to(self, other: "Point3D") -> float:
        """Manhattan distance to another point."""
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def midpoint(self, other: "Point3D") -> "Point3D":
        """Midpoint between this point and another."""
        return Point3D(
            (self.x + other.x) / 2,
            (self.y + other.y) / 2,
            (self.z + other.z) / 2
        )

    def dot(self, other: "Point3D") -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Point3D") -> "Point3D":
        """Cross product."""
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self) -> float:
        """Magnitude (distance from origin)."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Point3D":
        """Return unit vector in same direction."""
        mag = self.magnitude()
        if mag < 1e-10:
            return Point3D(0.0, 0.0, 0.0)
        return Point3D(self.x / mag, self.y / mag, self.z / mag)

    def angle_to(self, other: "Point3D") -> float:
        """Angle between this vector and another (radians)."""
        dot = self.dot(other)
        mags = self.magnitude() * other.magnitude()
        if mags < 1e-10:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / mags))
        return math.acos(cos_angle)

    def project_to_xy(self) -> Point2D:
        """Project to XY plane."""
        return Point2D(self.x, self.y)

    def project_to_xz(self) -> Point2D:
        """Project to XZ plane."""
        return Point2D(self.x, self.z)

    def project_to_yz(self) -> Point2D:
        """Project to YZ plane."""
        return Point2D(self.y, self.z)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return f"Point3D({self.x}, {self.y}, {self.z})"


class PointEncoder:
    """
    Encoder for 2D and 3D points.

    Encodes points using log-magnitude representation for scale invariance.
    Operations on embeddings preserve geometric relationships.
    """

    # Embedding layout - absolute positions
    X_SIGN_POS = 8
    X_LOG_POS = 9
    Y_SIGN_POS = 10
    Y_LOG_POS = 11
    Z_SIGN_POS = 12
    Z_LOG_POS = 13
    DIMENSION_POS = 14  # 2 or 3
    MAGNITUDE_POS = 15
    TYPE_START = 72

    def __init__(self):
        """Initialize the point encoder."""
        pass

    def encode(self, point: Union[Point2D, Point3D]) -> Any:
        """
        Encode a point as a 128-dimensional embedding.

        Uses log-magnitude representation for scale invariance.
        """
        backend = get_backend()
        embedding = backend.zeros((EMBEDDING_DIM,))

        is_3d = isinstance(point, Point3D)

        # Set domain tag
        tag_key = "geom_point3d" if is_3d else "geom_point2d"
        if tag_key in DOMAIN_TAGS:
            embedding = backend.at_add(embedding, slice(0, 8), DOMAIN_TAGS[tag_key])

        # Encode X coordinate
        x_sign, x_log = log_encode_value(point.x)
        embedding = backend.at_add(embedding, self.X_SIGN_POS, x_sign)
        embedding = backend.at_add(embedding, self.X_LOG_POS, x_log)

        # Encode Y coordinate
        y_sign, y_log = log_encode_value(point.y)
        embedding = backend.at_add(embedding, self.Y_SIGN_POS, y_sign)
        embedding = backend.at_add(embedding, self.Y_LOG_POS, y_log)

        # Encode Z coordinate (3D only)
        if is_3d:
            z_sign, z_log = log_encode_value(point.z)
            embedding = backend.at_add(embedding, self.Z_SIGN_POS, z_sign)
            embedding = backend.at_add(embedding, self.Z_LOG_POS, z_log)

        # Store dimensionality
        dim = 3.0 if is_3d else 2.0
        embedding = backend.at_add(embedding, self.DIMENSION_POS, dim)

        # Store magnitude for quick access
        mag = point.magnitude()
        if mag > 0:
            _, log_mag = log_encode_value(mag)
            embedding = backend.at_add(embedding, self.MAGNITUDE_POS, log_mag)

        return embedding

    def decode(self, embedding: Any) -> Union[Point2D, Point3D]:
        """
        Decode an embedding back to a Point.

        Returns Point2D or Point3D based on the stored dimensionality.
        """
        # Check dimensionality
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
            return Point3D(x, y, z)
        else:
            return Point2D(x, y)

    # Point operations

    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two points (vector addition).

        Decodes, adds, and re-encodes.
        """
        p1 = self.decode(emb1)
        p2 = self.decode(emb2)
        result = p1 + p2
        return self.encode(result)

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """
        Subtract two points (vector subtraction).

        Decodes, subtracts, and re-encodes.
        """
        p1 = self.decode(emb1)
        p2 = self.decode(emb2)
        result = p1 - p2
        return self.encode(result)

    def scale(self, embedding: Any, scalar: float) -> Any:
        """
        Scale a point by a scalar.
        """
        point = self.decode(embedding)
        result = point * scalar
        return self.encode(result)

    def negate(self, embedding: Any) -> Any:
        """Negate a point."""
        point = self.decode(embedding)
        return self.encode(-point)

    def distance(self, emb1: Any, emb2: Any) -> float:
        """
        Compute Euclidean distance between two points.

        Decodes and computes distance.
        """
        p1 = self.decode(emb1)
        p2 = self.decode(emb2)
        return p1.distance_to(p2)

    def midpoint(self, emb1: Any, emb2: Any) -> Any:
        """
        Compute midpoint between two points.

        Arithmetic midpoint.
        """
        p1 = self.decode(emb1)
        p2 = self.decode(emb2)
        return self.encode(p1.midpoint(p2))

    def dot_product(self, emb1: Any, emb2: Any) -> float:
        """
        Compute dot product of two points (as vectors).

        Algebraic dot product.
        """
        p1 = self.decode(emb1)
        p2 = self.decode(emb2)
        return p1.dot(p2)

    def magnitude(self, embedding: Any) -> float:
        """
        Compute magnitude (distance from origin).
        """
        point = self.decode(embedding)
        return point.magnitude()

    def normalize(self, embedding: Any) -> Any:
        """
        Normalize to unit vector.
        """
        point = self.decode(embedding)
        return self.encode(point.normalized())

    def angle_between(self, emb1: Any, emb2: Any) -> float:
        """
        Compute angle between two vectors (radians).

        Derived from dot product.
        """
        p1 = self.decode(emb1)
        p2 = self.decode(emb2)
        return p1.angle_to(p2)

    def is_2d(self, embedding: Any) -> bool:
        """Check if the point is 2D."""
        return int(float(embedding[self.DIMENSION_POS]) + 0.5) == 2

    def is_3d(self, embedding: Any) -> bool:
        """Check if the point is 3D."""
        return int(float(embedding[self.DIMENSION_POS]) + 0.5) == 3

    def equals(self, emb1: Any, emb2: Any, tolerance: float = 1e-9) -> bool:
        """
        Check if two point embeddings represent the same point.
        """
        p1 = self.decode(emb1)
        p2 = self.decode(emb2)
        return p1.distance_to(p2) < tolerance


# Utility functions

def origin_2d() -> Point2D:
    """Return the 2D origin."""
    return Point2D(0.0, 0.0)


def origin_3d() -> Point3D:
    """Return the 3D origin."""
    return Point3D(0.0, 0.0, 0.0)


def collinear(p1: Point2D, p2: Point2D, p3: Point2D, tolerance: float = 1e-9) -> bool:
    """
    Check if three 2D points are collinear.

    Uses cross product (area of triangle = 0).
    """
    # Vector from p1 to p2
    v1 = p2 - p1
    # Vector from p1 to p3
    v2 = p3 - p1
    # Cross product (z-component)
    cross = v1.cross(v2)
    return abs(cross) < tolerance


def coplanar(p1: Point3D, p2: Point3D, p3: Point3D, p4: Point3D, tolerance: float = 1e-9) -> bool:
    """
    Check if four 3D points are coplanar.

    Uses scalar triple product (volume of tetrahedron = 0).
    """
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    # Scalar triple product: v1 . (v2 x v3)
    triple = v1.dot(v2.cross(v3))
    return abs(triple) < tolerance
