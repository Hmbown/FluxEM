"""
Geometry Module for FluxEM-Domains

Provides embeddings for geometric objects and operations.
All operations preserve mathematical structure where possible.

Supports:
- Points in 2D and 3D space
- Vectors with component arithmetic
- Geometric transformations (translation, rotation, scaling, reflection)
- Shapes (triangles, rectangles, circles, polygons)
- Distances and angles
"""

from .points import Point2D, Point3D, PointEncoder
from .vectors import Vector2D, Vector3D, VectorEncoder
from .transforms import (
    Transform2D,
    Transform3D,
    TransformEncoder,
    translate,
    rotate,
    scale,
    reflect,
)
from .shapes import (
    Triangle,
    Rectangle,
    Circle,
    Polygon,
    ShapeEncoder,
    ShapeType,
)

__all__ = [
    # Points
    "Point2D",
    "Point3D",
    "PointEncoder",
    # Vectors
    "Vector2D",
    "Vector3D",
    "VectorEncoder",
    # Transforms
    "Transform2D",
    "Transform3D",
    "TransformEncoder",
    "translate",
    "rotate",
    "scale",
    "reflect",
    # Shapes
    "Triangle",
    "Rectangle",
    "Circle",
    "Polygon",
    "ShapeEncoder",
    "ShapeType",
]
