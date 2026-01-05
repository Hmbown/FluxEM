"""
Geometric Transformations for FluxEM-Domains

Provides 2D and 3D transformations using matrix representation.
All transformations are composable.

Transformations supported:
- Translation
- Rotation (around origin or arbitrary point/axis)
- Scaling (uniform and non-uniform)
- Reflection
- Shearing

Embedding Layout (128 dimensions):
  - dims 0-7:   Domain tag (geom_transform2d or geom_transform3d)
  - dims 8-17:  2D: 3x3 affine matrix (row-major, 9 elements + padding)
                3D: First 10 of 4x4 matrix
  - dims 18-32: 3D: Remaining 6 elements of 4x4 matrix + metadata
  - dims 72-79: Transform type encoding
  - dims 80-95: Shared semantic features
  - dims 96-127: Cross-domain features
"""

from dataclasses import dataclass
from typing import Any, Union, List, Tuple, Optional
import math
from ...backend import get_backend

from ...core.base import DOMAIN_TAGS, EMBEDDING_DIM
from .points import Point2D, Point3D
from .vectors import Vector2D, Vector3D


@dataclass
class Transform2D:
    """
    2D affine transformation represented as a 3x3 matrix.

    The matrix is stored in the form:
    | a  b  tx |
    | c  d  ty |
    | 0  0  1  |

    Where:
    - (a, b, c, d) is the linear transformation part
    - (tx, ty) is the translation part
    """
    # Linear transformation part
    a: float
    b: float
    c: float
    d: float
    # Translation part
    tx: float
    ty: float

    @classmethod
    def identity(cls) -> "Transform2D":
        """Create identity transform."""
        return cls(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    @classmethod
    def translation(cls, dx: float, dy: float) -> "Transform2D":
        """Create translation transform."""
        return cls(1.0, 0.0, 0.0, 1.0, dx, dy)

    @classmethod
    def rotation(cls, angle: float, center: Optional[Point2D] = None) -> "Transform2D":
        """
        Create rotation transform around a point.

        Args:
            angle: Rotation angle in radians (counter-clockwise)
            center: Center of rotation (default: origin)
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        if center is None or (center.x == 0 and center.y == 0):
            return cls(cos_a, -sin_a, sin_a, cos_a, 0.0, 0.0)
        else:
            # Translate to origin, rotate, translate back
            # tx = cx - cx*cos + cy*sin
            # ty = cy - cx*sin - cy*cos
            tx = center.x - center.x * cos_a + center.y * sin_a
            ty = center.y - center.x * sin_a - center.y * cos_a
            return cls(cos_a, -sin_a, sin_a, cos_a, tx, ty)

    @classmethod
    def scaling(cls, sx: float, sy: Optional[float] = None, center: Optional[Point2D] = None) -> "Transform2D":
        """
        Create scaling transform.

        Args:
            sx: X scale factor
            sy: Y scale factor (default: same as sx for uniform scaling)
            center: Center of scaling (default: origin)
        """
        if sy is None:
            sy = sx

        if center is None or (center.x == 0 and center.y == 0):
            return cls(sx, 0.0, 0.0, sy, 0.0, 0.0)
        else:
            tx = center.x * (1 - sx)
            ty = center.y * (1 - sy)
            return cls(sx, 0.0, 0.0, sy, tx, ty)

    @classmethod
    def reflection_x(cls) -> "Transform2D":
        """Reflect across the x-axis."""
        return cls(1.0, 0.0, 0.0, -1.0, 0.0, 0.0)

    @classmethod
    def reflection_y(cls) -> "Transform2D":
        """Reflect across the y-axis."""
        return cls(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    @classmethod
    def reflection_line(cls, angle: float) -> "Transform2D":
        """Reflect across a line through origin at given angle."""
        cos_2a = math.cos(2 * angle)
        sin_2a = math.sin(2 * angle)
        return cls(cos_2a, sin_2a, sin_2a, -cos_2a, 0.0, 0.0)

    @classmethod
    def shear(cls, shx: float, shy: float) -> "Transform2D":
        """Create shearing transform."""
        return cls(1.0, shx, shy, 1.0, 0.0, 0.0)

    def apply(self, point: Point2D) -> Point2D:
        """Apply transform to a point via matrix multiplication."""
        new_x = self.a * point.x + self.b * point.y + self.tx
        new_y = self.c * point.x + self.d * point.y + self.ty
        return Point2D(new_x, new_y)

    def apply_vector(self, vector: Vector2D) -> Vector2D:
        """Apply transform to a vector (ignores translation)."""
        new_x = self.a * vector.x + self.b * vector.y
        new_y = self.c * vector.x + self.d * vector.y
        return Vector2D(new_x, new_y)

    def compose(self, other: "Transform2D") -> "Transform2D":
        """
        Compose with another transform.

        Returns self * other (apply other first, then self).
        Matrix multiplication.
        """
        return Transform2D(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
            tx=self.a * other.tx + self.b * other.ty + self.tx,
            ty=self.c * other.tx + self.d * other.ty + self.ty,
        )

    def inverse(self) -> "Transform2D":
        """
        Compute inverse transform.

        Matrix inversion for affine transforms.
        """
        det = self.a * self.d - self.b * self.c
        if abs(det) < 1e-10:
            raise ValueError("Transform is not invertible (determinant is zero)")

        inv_det = 1.0 / det
        return Transform2D(
            a=self.d * inv_det,
            b=-self.b * inv_det,
            c=-self.c * inv_det,
            d=self.a * inv_det,
            tx=(self.b * self.ty - self.d * self.tx) * inv_det,
            ty=(self.c * self.tx - self.a * self.ty) * inv_det,
        )

    def determinant(self) -> float:
        """Compute determinant of the linear part."""
        return self.a * self.d - self.b * self.c

    def is_rigid(self, tolerance: float = 1e-9) -> bool:
        """Check if transform preserves distances (rotation + translation only)."""
        det = self.determinant()
        return abs(abs(det) - 1.0) < tolerance

    def to_matrix(self) -> List[List[float]]:
        """Convert to 3x3 matrix."""
        return [
            [self.a, self.b, self.tx],
            [self.c, self.d, self.ty],
            [0.0, 0.0, 1.0]
        ]

    def __repr__(self) -> str:
        return f"Transform2D(a={self.a}, b={self.b}, c={self.c}, d={self.d}, tx={self.tx}, ty={self.ty})"


@dataclass
class Transform3D:
    """
    3D affine transformation represented as a 4x4 matrix.

    The matrix is stored in the form:
    | m00 m01 m02 tx |
    | m10 m11 m12 ty |
    | m20 m21 m22 tz |
    | 0   0   0   1  |
    """
    # 3x3 linear transformation part
    m00: float
    m01: float
    m02: float
    m10: float
    m11: float
    m12: float
    m20: float
    m21: float
    m22: float
    # Translation part
    tx: float
    ty: float
    tz: float

    @classmethod
    def identity(cls) -> "Transform3D":
        """Create identity transform."""
        return cls(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)

    @classmethod
    def translation(cls, dx: float, dy: float, dz: float) -> "Transform3D":
        """Create translation transform."""
        return cls(1, 0, 0, 0, 1, 0, 0, 0, 1, dx, dy, dz)

    @classmethod
    def rotation_x(cls, angle: float) -> "Transform3D":
        """Create rotation around X axis."""
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(1, 0, 0, 0, c, -s, 0, s, c, 0, 0, 0)

    @classmethod
    def rotation_y(cls, angle: float) -> "Transform3D":
        """Create rotation around Y axis."""
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(c, 0, s, 0, 1, 0, -s, 0, c, 0, 0, 0)

    @classmethod
    def rotation_z(cls, angle: float) -> "Transform3D":
        """Create rotation around Z axis."""
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(c, -s, 0, s, c, 0, 0, 0, 1, 0, 0, 0)

    @classmethod
    def rotation_axis(cls, axis: Vector3D, angle: float) -> "Transform3D":
        """
        Create rotation around arbitrary axis (Rodrigues' formula).

        Uses Rodrigues' formula for the given angle.
        """
        k = axis.normalized()
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c

        return cls(
            m00=t * k.x * k.x + c,
            m01=t * k.x * k.y - s * k.z,
            m02=t * k.x * k.z + s * k.y,
            m10=t * k.x * k.y + s * k.z,
            m11=t * k.y * k.y + c,
            m12=t * k.y * k.z - s * k.x,
            m20=t * k.x * k.z - s * k.y,
            m21=t * k.y * k.z + s * k.x,
            m22=t * k.z * k.z + c,
            tx=0, ty=0, tz=0
        )

    @classmethod
    def scaling(cls, sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> "Transform3D":
        """Create scaling transform."""
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        return cls(sx, 0, 0, 0, sy, 0, 0, 0, sz, 0, 0, 0)

    @classmethod
    def reflection_xy(cls) -> "Transform3D":
        """Reflect across XY plane (negate z)."""
        return cls(1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0)

    @classmethod
    def reflection_xz(cls) -> "Transform3D":
        """Reflect across XZ plane (negate y)."""
        return cls(1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0)

    @classmethod
    def reflection_yz(cls) -> "Transform3D":
        """Reflect across YZ plane (negate x)."""
        return cls(-1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)

    def apply(self, point: Point3D) -> Point3D:
        """Apply transform to a point."""
        new_x = self.m00 * point.x + self.m01 * point.y + self.m02 * point.z + self.tx
        new_y = self.m10 * point.x + self.m11 * point.y + self.m12 * point.z + self.ty
        new_z = self.m20 * point.x + self.m21 * point.y + self.m22 * point.z + self.tz
        return Point3D(new_x, new_y, new_z)

    def apply_vector(self, vector: Vector3D) -> Vector3D:
        """Apply transform to a vector (ignores translation)."""
        new_x = self.m00 * vector.x + self.m01 * vector.y + self.m02 * vector.z
        new_y = self.m10 * vector.x + self.m11 * vector.y + self.m12 * vector.z
        new_z = self.m20 * vector.x + self.m21 * vector.y + self.m22 * vector.z
        return Vector3D(new_x, new_y, new_z)

    def compose(self, other: "Transform3D") -> "Transform3D":
        """
        Compose with another transform.

        Returns self * other (apply other first, then self).
        Matrix multiplication.
        """
        return Transform3D(
            m00=self.m00 * other.m00 + self.m01 * other.m10 + self.m02 * other.m20,
            m01=self.m00 * other.m01 + self.m01 * other.m11 + self.m02 * other.m21,
            m02=self.m00 * other.m02 + self.m01 * other.m12 + self.m02 * other.m22,
            m10=self.m10 * other.m00 + self.m11 * other.m10 + self.m12 * other.m20,
            m11=self.m10 * other.m01 + self.m11 * other.m11 + self.m12 * other.m21,
            m12=self.m10 * other.m02 + self.m11 * other.m12 + self.m12 * other.m22,
            m20=self.m20 * other.m00 + self.m21 * other.m10 + self.m22 * other.m20,
            m21=self.m20 * other.m01 + self.m21 * other.m11 + self.m22 * other.m21,
            m22=self.m20 * other.m02 + self.m21 * other.m12 + self.m22 * other.m22,
            tx=self.m00 * other.tx + self.m01 * other.ty + self.m02 * other.tz + self.tx,
            ty=self.m10 * other.tx + self.m11 * other.ty + self.m12 * other.tz + self.ty,
            tz=self.m20 * other.tx + self.m21 * other.ty + self.m22 * other.tz + self.tz,
        )

    def determinant(self) -> float:
        """Compute determinant of the linear part."""
        return (
            self.m00 * (self.m11 * self.m22 - self.m12 * self.m21)
            - self.m01 * (self.m10 * self.m22 - self.m12 * self.m20)
            + self.m02 * (self.m10 * self.m21 - self.m11 * self.m20)
        )

    def inverse(self) -> "Transform3D":
        """Compute inverse transform."""
        det = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("Transform is not invertible")

        inv_det = 1.0 / det

        # Inverse of 3x3 linear part
        inv_m00 = (self.m11 * self.m22 - self.m12 * self.m21) * inv_det
        inv_m01 = (self.m02 * self.m21 - self.m01 * self.m22) * inv_det
        inv_m02 = (self.m01 * self.m12 - self.m02 * self.m11) * inv_det
        inv_m10 = (self.m12 * self.m20 - self.m10 * self.m22) * inv_det
        inv_m11 = (self.m00 * self.m22 - self.m02 * self.m20) * inv_det
        inv_m12 = (self.m02 * self.m10 - self.m00 * self.m12) * inv_det
        inv_m20 = (self.m10 * self.m21 - self.m11 * self.m20) * inv_det
        inv_m21 = (self.m01 * self.m20 - self.m00 * self.m21) * inv_det
        inv_m22 = (self.m00 * self.m11 - self.m01 * self.m10) * inv_det

        # Inverse translation
        inv_tx = -(inv_m00 * self.tx + inv_m01 * self.ty + inv_m02 * self.tz)
        inv_ty = -(inv_m10 * self.tx + inv_m11 * self.ty + inv_m12 * self.tz)
        inv_tz = -(inv_m20 * self.tx + inv_m21 * self.ty + inv_m22 * self.tz)

        return Transform3D(
            inv_m00, inv_m01, inv_m02,
            inv_m10, inv_m11, inv_m12,
            inv_m20, inv_m21, inv_m22,
            inv_tx, inv_ty, inv_tz
        )

    def is_rigid(self, tolerance: float = 1e-9) -> bool:
        """Check if transform preserves distances."""
        det = self.determinant()
        return abs(abs(det) - 1.0) < tolerance

    def __repr__(self) -> str:
        return f"Transform3D(matrix={self.to_matrix()[:3]})"

    def to_matrix(self) -> List[List[float]]:
        """Convert to 4x4 matrix."""
        return [
            [self.m00, self.m01, self.m02, self.tx],
            [self.m10, self.m11, self.m12, self.ty],
            [self.m20, self.m21, self.m22, self.tz],
            [0.0, 0.0, 0.0, 1.0]
        ]


class TransformEncoder:
    """
    Encoder for 2D and 3D transforms.

    Stores transformation matrix elements directly in embedding.
    """

    # Embedding layout
    MATRIX_START = 8
    MATRIX_SIZE_2D = 6  # a, b, c, d, tx, ty
    MATRIX_SIZE_3D = 12  # 9 rotation + 3 translation
    DIMENSION_POS = 20
    TYPE_START = 72

    def __init__(self):
        """Initialize the transform encoder."""
        pass

    def encode(self, transform: Union[Transform2D, Transform3D]) -> Any:
        """Encode a transform as a 128-dimensional embedding."""
        backend = get_backend()
        embedding = backend.zeros((EMBEDDING_DIM,))

        is_3d = isinstance(transform, Transform3D)

        # Set domain tag
        tag_key = "geom_transform3d" if is_3d else "geom_transform2d"
        if tag_key in DOMAIN_TAGS:
            embedding = backend.at_add(embedding, slice(0, 8), DOMAIN_TAGS[tag_key])

        if is_3d:
            # Store 3D matrix elements
            elements = [
                transform.m00, transform.m01, transform.m02,
                transform.m10, transform.m11, transform.m12,
                transform.m20, transform.m21, transform.m22,
                transform.tx, transform.ty, transform.tz
            ]
            for i, val in enumerate(elements):
                embedding = backend.at_add(embedding, self.MATRIX_START + i, val)
            embedding = backend.at_add(embedding, self.DIMENSION_POS, 3.0)
        else:
            # Store 2D matrix elements
            elements = [
                transform.a, transform.b, transform.c, transform.d,
                transform.tx, transform.ty
            ]
            for i, val in enumerate(elements):
                embedding = backend.at_add(embedding, self.MATRIX_START + i, val)
            embedding = backend.at_add(embedding, self.DIMENSION_POS, 2.0)

        return embedding

    def decode(self, embedding: Any) -> Union[Transform2D, Transform3D]:
        """Decode an embedding back to a Transform."""
        dim = int(float(embedding[self.DIMENSION_POS]) + 0.5)

        if dim == 3:
            return Transform3D(
                m00=float(embedding[self.MATRIX_START + 0]),
                m01=float(embedding[self.MATRIX_START + 1]),
                m02=float(embedding[self.MATRIX_START + 2]),
                m10=float(embedding[self.MATRIX_START + 3]),
                m11=float(embedding[self.MATRIX_START + 4]),
                m12=float(embedding[self.MATRIX_START + 5]),
                m20=float(embedding[self.MATRIX_START + 6]),
                m21=float(embedding[self.MATRIX_START + 7]),
                m22=float(embedding[self.MATRIX_START + 8]),
                tx=float(embedding[self.MATRIX_START + 9]),
                ty=float(embedding[self.MATRIX_START + 10]),
                tz=float(embedding[self.MATRIX_START + 11]),
            )
        else:
            return Transform2D(
                a=float(embedding[self.MATRIX_START + 0]),
                b=float(embedding[self.MATRIX_START + 1]),
                c=float(embedding[self.MATRIX_START + 2]),
                d=float(embedding[self.MATRIX_START + 3]),
                tx=float(embedding[self.MATRIX_START + 4]),
                ty=float(embedding[self.MATRIX_START + 5]),
            )

    def compose(self, emb1: Any, emb2: Any) -> Any:
        """Compose two transforms via matrix multiplication."""
        t1 = self.decode(emb1)
        t2 = self.decode(emb2)
        return self.encode(t1.compose(t2))

    def inverse(self, embedding: Any) -> Any:
        """Compute inverse transform."""
        t = self.decode(embedding)
        return self.encode(t.inverse())


# Convenience functions

def translate(dx: float, dy: float, dz: Optional[float] = None) -> Union[Transform2D, Transform3D]:
    """Create a translation transform."""
    if dz is None:
        return Transform2D.translation(dx, dy)
    else:
        return Transform3D.translation(dx, dy, dz)


def rotate(angle: float, axis: Optional[Vector3D] = None, center: Optional[Union[Point2D, Point3D]] = None) -> Union[Transform2D, Transform3D]:
    """
    Create a rotation transform.

    For 2D: rotates around a point (default: origin)
    For 3D: rotates around an axis (default: z-axis) at origin
    """
    if axis is not None:
        return Transform3D.rotation_axis(axis, angle)
    elif isinstance(center, Point3D):
        # 3D rotation around z-axis at a point
        t = Transform3D.translation(-center.x, -center.y, -center.z)
        r = Transform3D.rotation_z(angle)
        t_back = Transform3D.translation(center.x, center.y, center.z)
        return t_back.compose(r.compose(t))
    else:
        return Transform2D.rotation(angle, center)


def scale(sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> Union[Transform2D, Transform3D]:
    """Create a scaling transform."""
    if sz is not None:
        return Transform3D.scaling(sx, sy, sz)
    else:
        return Transform2D.scaling(sx, sy)


def reflect(axis_or_plane: str) -> Union[Transform2D, Transform3D]:
    """
    Create a reflection transform.

    For 2D: 'x' or 'y' for axis reflection
    For 3D: 'xy', 'xz', or 'yz' for plane reflection
    """
    if axis_or_plane == 'x':
        return Transform2D.reflection_x()
    elif axis_or_plane == 'y':
        return Transform2D.reflection_y()
    elif axis_or_plane == 'xy':
        return Transform3D.reflection_xy()
    elif axis_or_plane == 'xz':
        return Transform3D.reflection_xz()
    elif axis_or_plane == 'yz':
        return Transform3D.reflection_yz()
    else:
        raise ValueError(f"Unknown axis/plane: {axis_or_plane}")
