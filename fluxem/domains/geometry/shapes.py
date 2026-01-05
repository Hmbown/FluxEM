"""
Geometric Shapes for FluxEM-Domains

Provides 2D shapes with geometric properties.
All area, perimeter, and property calculations use closed-form formulas.

Supported shapes:
- Triangle: defined by 3 vertices
- Rectangle: defined by center, width, height, rotation
- Circle: defined by center and radius
- Polygon: defined by vertices (convex or general)

Embedding Layout (128 dimensions):
  - dims 0-7:   Domain tag (geom_triangle, geom_rectangle, geom_circle, geom_polygon)
  - dims 8-31:  Shape-specific data (vertices, dimensions)
  - dims 32-39: Cached area, perimeter
  - dims 40-47: Shape properties (convex, regular, etc.)
  - dims 72-79: Shape type encoding
  - dims 80-95: Shared semantic features
  - dims 96-127: Cross-domain features
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Tuple, Optional
import math
from ...backend import get_backend

from ...core.base import DOMAIN_TAGS, EMBEDDING_DIM, log_encode_value, log_decode_value
from .points import Point2D


class ShapeType(Enum):
    """Types of geometric shapes."""
    TRIANGLE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()
    POLYGON = auto()
    ELLIPSE = auto()


@dataclass(frozen=True)
class Triangle:
    """
    Triangle defined by three vertices.

    All geometric properties are computed from vertices.
    """
    p1: Point2D
    p2: Point2D
    p3: Point2D

    def vertices(self) -> Tuple[Point2D, Point2D, Point2D]:
        """Return the three vertices."""
        return (self.p1, self.p2, self.p3)

    def sides(self) -> Tuple[float, float, float]:
        """Return the lengths of the three sides."""
        a = self.p2.distance_to(self.p3)
        b = self.p1.distance_to(self.p3)
        c = self.p1.distance_to(self.p2)
        return (a, b, c)

    def perimeter(self) -> float:
        """Compute perimeter. sum of side lengths."""
        a, b, c = self.sides()
        return a + b + c

    def area(self) -> float:
        """
        Compute area using the shoelace formula.

        |det([p2-p1, p3-p1])| / 2
        """
        return abs(
            (self.p2.x - self.p1.x) * (self.p3.y - self.p1.y)
            - (self.p3.x - self.p1.x) * (self.p2.y - self.p1.y)
        ) / 2

    def centroid(self) -> Point2D:
        """Compute centroid (center of mass). average of vertices."""
        return Point2D(
            (self.p1.x + self.p2.x + self.p3.x) / 3,
            (self.p1.y + self.p2.y + self.p3.y) / 3
        )

    def incenter(self) -> Point2D:
        """Compute incenter (center of inscribed circle)."""
        a, b, c = self.sides()
        s = a + b + c
        return Point2D(
            (a * self.p1.x + b * self.p2.x + c * self.p3.x) / s,
            (a * self.p1.y + b * self.p2.y + c * self.p3.y) / s
        )

    def circumcenter(self) -> Point2D:
        """Compute circumcenter (center of circumscribed circle)."""
        ax, ay = self.p1.x, self.p1.y
        bx, by = self.p2.x, self.p2.y
        cx, cy = self.p3.x, self.p3.y

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return self.centroid()  # Degenerate case

        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

        return Point2D(ux, uy)

    def inradius(self) -> float:
        """Radius of inscribed circle. area / semi-perimeter."""
        return self.area() / (self.perimeter() / 2)

    def circumradius(self) -> float:
        """Radius of circumscribed circle computed from formula."""
        a, b, c = self.sides()
        area = self.area()
        if area < 1e-10:
            return 0.0
        return (a * b * c) / (4 * area)

    def angles(self) -> Tuple[float, float, float]:
        """Compute interior angles in radians from dot products."""
        v1 = self.p2 - self.p1
        v2 = self.p3 - self.p1
        v3 = self.p1 - self.p2
        v4 = self.p3 - self.p2
        v5 = self.p1 - self.p3
        v6 = self.p2 - self.p3

        angle1 = v1.angle_to(v2)
        angle2 = v3.angle_to(v4)
        angle3 = v5.angle_to(v6)

        return (angle1, angle2, angle3)

    def is_right(self, tolerance: float = 1e-9) -> bool:
        """Check if triangle has a right angle."""
        angles = self.angles()
        right_angle = math.pi / 2
        return any(abs(a - right_angle) < tolerance for a in angles)

    def is_isosceles(self, tolerance: float = 1e-9) -> bool:
        """Check if triangle has two equal sides."""
        a, b, c = self.sides()
        return (abs(a - b) < tolerance or abs(b - c) < tolerance or abs(a - c) < tolerance)

    def is_equilateral(self, tolerance: float = 1e-9) -> bool:
        """Check if all sides are equal."""
        a, b, c = self.sides()
        return abs(a - b) < tolerance and abs(b - c) < tolerance

    def is_acute(self) -> bool:
        """Check if all angles are acute."""
        angles = self.angles()
        return all(a < math.pi / 2 for a in angles)

    def is_obtuse(self) -> bool:
        """Check if one angle is obtuse."""
        angles = self.angles()
        return any(a > math.pi / 2 for a in angles)

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside triangle using barycentric coordinates."""
        v0 = self.p3 - self.p1
        v1 = self.p2 - self.p1
        v2 = point - self.p1

        dot00 = v0.dot(v0)
        dot01 = v0.dot(v1)
        dot02 = v0.dot(v2)
        dot11 = v1.dot(v1)
        dot12 = v1.dot(v2)

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            return False

        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom

        return (u >= 0) and (v >= 0) and (u + v <= 1)


@dataclass(frozen=True)
class Rectangle:
    """
    Axis-aligned or rotated rectangle.

    Defined by center, width, height, and rotation angle.
    """
    center: Point2D
    width: float
    height: float
    rotation: float = 0.0  # Rotation angle in radians

    def vertices(self) -> Tuple[Point2D, Point2D, Point2D, Point2D]:
        """Return the four corners."""
        hw, hh = self.width / 2, self.height / 2
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)

        # Local corners before rotation
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

        result = []
        for lx, ly in corners:
            # Rotate and translate
            x = lx * cos_r - ly * sin_r + self.center.x
            y = lx * sin_r + ly * cos_r + self.center.y
            result.append(Point2D(x, y))

        return tuple(result)

    def perimeter(self) -> float:
        """Compute perimeter. 2*(width + height)."""
        return 2 * (self.width + self.height)

    def area(self) -> float:
        """Compute area. width * height."""
        return self.width * self.height

    def diagonal(self) -> float:
        """Length of diagonal. sqrt(w² + h²)."""
        return math.sqrt(self.width ** 2 + self.height ** 2)

    def is_square(self, tolerance: float = 1e-9) -> bool:
        """Check if rectangle is a square."""
        return abs(self.width - self.height) < tolerance

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside rectangle."""
        # Transform point to local coordinates
        dx = point.x - self.center.x
        dy = point.y - self.center.y

        cos_r = math.cos(-self.rotation)
        sin_r = math.sin(-self.rotation)

        local_x = dx * cos_r - dy * sin_r
        local_y = dx * sin_r + dy * cos_r

        return abs(local_x) <= self.width / 2 and abs(local_y) <= self.height / 2


@dataclass(frozen=True)
class Circle:
    """
    Circle defined by center and radius.

    All properties are computed from center and radius.
    """
    center: Point2D
    radius: float

    def perimeter(self) -> float:
        """Compute circumference. 2*pi*r."""
        return 2 * math.pi * self.radius

    def area(self) -> float:
        """Compute area. pi*r²."""
        return math.pi * self.radius ** 2

    def diameter(self) -> float:
        """Compute diameter. 2*r."""
        return 2 * self.radius

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside circle."""
        return self.center.distance_to(point) <= self.radius

    def intersects(self, other: "Circle") -> bool:
        """Check if two circles intersect."""
        dist = self.center.distance_to(other.center)
        return dist <= self.radius + other.radius

    def tangent_points(self, other: "Circle") -> Optional[Tuple[Point2D, Point2D]]:
        """Find tangent points between two circles if they touch externally."""
        dist = self.center.distance_to(other.center)
        if abs(dist - (self.radius + other.radius)) > 1e-9:
            return None

        # Point of tangency
        t = self.radius / dist
        x = self.center.x + t * (other.center.x - self.center.x)
        y = self.center.y + t * (other.center.y - self.center.y)
        return (Point2D(x, y), Point2D(x, y))


@dataclass(frozen=True)
class Polygon:
    """
    General polygon defined by vertices.

    Vertices should be in counter-clockwise order for positive area.
    """
    vertices: Tuple[Point2D, ...]

    def __post_init__(self):
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

    @classmethod
    def regular(cls, n: int, radius: float, center: Point2D = Point2D(0, 0)) -> "Polygon":
        """Create a regular n-gon inscribed in a circle."""
        vertices = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            vertices.append(Point2D(x, y))
        return cls(tuple(vertices))

    def num_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    def edges(self) -> List[Tuple[Point2D, Point2D]]:
        """Return list of edges as (start, end) tuples."""
        result = []
        n = len(self.vertices)
        for i in range(n):
            result.append((self.vertices[i], self.vertices[(i + 1) % n]))
        return result

    def perimeter(self) -> float:
        """Compute perimeter. sum of edge lengths."""
        total = 0.0
        for p1, p2 in self.edges():
            total += p1.distance_to(p2)
        return total

    def area(self) -> float:
        """
        Compute area using shoelace formula.

        Sum of signed areas of triangles from origin.
        """
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2

    def centroid(self) -> Point2D:
        """Compute centroid for simple polygons."""
        n = len(self.vertices)
        area = self.area()
        if area < 1e-10:
            # Degenerate: return average
            x = sum(v.x for v in self.vertices) / n
            y = sum(v.y for v in self.vertices) / n
            return Point2D(x, y)

        cx, cy = 0.0, 0.0
        for i in range(n):
            j = (i + 1) % n
            cross = self.vertices[i].x * self.vertices[j].y - self.vertices[j].x * self.vertices[i].y
            cx += (self.vertices[i].x + self.vertices[j].x) * cross
            cy += (self.vertices[i].y + self.vertices[j].y) * cross

        factor = 1 / (6 * area)
        return Point2D(cx * factor, cy * factor)

    def is_convex(self) -> bool:
        """Check if polygon is convex. all cross products same sign."""
        n = len(self.vertices)
        if n < 3:
            return False

        sign = None
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            p3 = self.vertices[(i + 2) % n]

            v1 = p2 - p1
            v2 = p3 - p2
            cross = v1.cross(v2)

            if abs(cross) > 1e-10:
                if sign is None:
                    sign = cross > 0
                elif (cross > 0) != sign:
                    return False

        return True

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(self.vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i].x, self.vertices[i].y
            xj, yj = self.vertices[j].x, self.vertices[j].y

            if ((yi > point.y) != (yj > point.y)) and \
               (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside


class ShapeEncoder:
    """
    Encoder for geometric shapes.

    Stores shape type and key parameters in embedding.
    """

    # Embedding layout
    SHAPE_TYPE_POS = 8
    PARAM_START = 9
    AREA_POS = 32
    PERIMETER_POS = 34
    PROPERTY_FLAGS_START = 40

    def __init__(self):
        pass

    def encode(self, shape) -> Any:
        """Encode a shape as a 128-dimensional embedding."""
        backend = get_backend()
        embedding = backend.zeros((EMBEDDING_DIM,))

        if isinstance(shape, Triangle):
            embedding = self._encode_triangle(embedding, shape)
        elif isinstance(shape, Rectangle):
            embedding = self._encode_rectangle(embedding, shape)
        elif isinstance(shape, Circle):
            embedding = self._encode_circle(embedding, shape)
        elif isinstance(shape, Polygon):
            embedding = self._encode_polygon(embedding, shape)
        else:
            raise ValueError(f"Unknown shape type: {type(shape)}")

        return embedding

    def _encode_triangle(self, embedding: Any, tri: Triangle) -> Any:
        backend = get_backend()
        if "geom_triangle" in DOMAIN_TAGS:
            embedding = backend.at_add(embedding, slice(0, 8), DOMAIN_TAGS["geom_triangle"])

        embedding = backend.at_add(embedding, self.SHAPE_TYPE_POS, float(ShapeType.TRIANGLE.value))

        # Store vertices
        for i, p in enumerate([tri.p1, tri.p2, tri.p3]):
            embedding = backend.at_add(embedding, self.PARAM_START + i * 2, p.x)
            embedding = backend.at_add(embedding, self.PARAM_START + i * 2 + 1, p.y)

        # Store area and perimeter
        _, area_log = log_encode_value(tri.area())
        embedding = backend.at_add(embedding, self.AREA_POS, 1.0)  # sign
        embedding = backend.at_add(embedding, self.AREA_POS + 1, area_log)

        _, perim_log = log_encode_value(tri.perimeter())
        embedding = backend.at_add(embedding, self.PERIMETER_POS, 1.0)
        embedding = backend.at_add(embedding, self.PERIMETER_POS + 1, perim_log)

        # Property flags
        if tri.is_equilateral():
            embedding = backend.at_add(embedding, self.PROPERTY_FLAGS_START, 1.0)
        if tri.is_isosceles():
            embedding = backend.at_add(embedding, self.PROPERTY_FLAGS_START + 1, 1.0)
        if tri.is_right():
            embedding = backend.at_add(embedding, self.PROPERTY_FLAGS_START + 2, 1.0)

        return embedding

    def _encode_rectangle(self, embedding: Any, rect: Rectangle) -> Any:
        backend = get_backend()
        if "geom_rectangle" in DOMAIN_TAGS:
            embedding = backend.at_add(embedding, slice(0, 8), DOMAIN_TAGS["geom_rectangle"])

        embedding = backend.at_add(embedding, self.SHAPE_TYPE_POS, float(ShapeType.RECTANGLE.value))

        # Store center, width, height, rotation
        embedding = backend.at_add(embedding, self.PARAM_START, rect.center.x)
        embedding = backend.at_add(embedding, self.PARAM_START + 1, rect.center.y)
        embedding = backend.at_add(embedding, self.PARAM_START + 2, rect.width)
        embedding = backend.at_add(embedding, self.PARAM_START + 3, rect.height)
        embedding = backend.at_add(embedding, self.PARAM_START + 4, rect.rotation)

        # Store area and perimeter
        _, area_log = log_encode_value(rect.area())
        embedding = backend.at_add(embedding, self.AREA_POS, 1.0)
        embedding = backend.at_add(embedding, self.AREA_POS + 1, area_log)

        _, perim_log = log_encode_value(rect.perimeter())
        embedding = backend.at_add(embedding, self.PERIMETER_POS, 1.0)
        embedding = backend.at_add(embedding, self.PERIMETER_POS + 1, perim_log)

        # Property flags
        if rect.is_square():
            embedding = backend.at_add(embedding, self.PROPERTY_FLAGS_START, 1.0)

        return embedding

    def _encode_circle(self, embedding: Any, circle: Circle) -> Any:
        backend = get_backend()
        if "geom_circle" in DOMAIN_TAGS:
            embedding = backend.at_add(embedding, slice(0, 8), DOMAIN_TAGS["geom_circle"])

        embedding = backend.at_add(embedding, self.SHAPE_TYPE_POS, float(ShapeType.CIRCLE.value))

        # Store center and radius
        embedding = backend.at_add(embedding, self.PARAM_START, circle.center.x)
        embedding = backend.at_add(embedding, self.PARAM_START + 1, circle.center.y)
        embedding = backend.at_add(embedding, self.PARAM_START + 2, circle.radius)

        # Store area and perimeter
        _, area_log = log_encode_value(circle.area())
        embedding = backend.at_add(embedding, self.AREA_POS, 1.0)
        embedding = backend.at_add(embedding, self.AREA_POS + 1, area_log)

        _, perim_log = log_encode_value(circle.perimeter())
        embedding = backend.at_add(embedding, self.PERIMETER_POS, 1.0)
        embedding = backend.at_add(embedding, self.PERIMETER_POS + 1, perim_log)

        return embedding

    def _encode_polygon(self, embedding: Any, poly: Polygon) -> Any:
        backend = get_backend()
        if "geom_polygon" in DOMAIN_TAGS:
            embedding = backend.at_add(embedding, slice(0, 8), DOMAIN_TAGS["geom_polygon"])

        embedding = backend.at_add(embedding, self.SHAPE_TYPE_POS, float(ShapeType.POLYGON.value))

        # Store number of vertices and first few vertices
        n = len(poly.vertices)
        embedding = backend.at_add(embedding, self.PARAM_START, float(n))

        # Store up to 10 vertices
        for i, v in enumerate(poly.vertices[:10]):
            embedding = backend.at_add(embedding, self.PARAM_START + 1 + i * 2, v.x)
            embedding = backend.at_add(embedding, self.PARAM_START + 2 + i * 2, v.y)

        # Store area and perimeter
        _, area_log = log_encode_value(poly.area())
        embedding = backend.at_add(embedding, self.AREA_POS, 1.0)
        embedding = backend.at_add(embedding, self.AREA_POS + 1, area_log)

        _, perim_log = log_encode_value(poly.perimeter())
        embedding = backend.at_add(embedding, self.PERIMETER_POS, 1.0)
        embedding = backend.at_add(embedding, self.PERIMETER_POS + 1, perim_log)

        # Property flags
        if poly.is_convex():
            embedding = backend.at_add(embedding, self.PROPERTY_FLAGS_START, 1.0)

        return embedding

    def get_area(self, embedding: Any) -> float:
        """Extract area from embedding."""
        sign = float(embedding[self.AREA_POS])
        log_val = float(embedding[self.AREA_POS + 1])
        return log_decode_value(sign, log_val)

    def get_perimeter(self, embedding: Any) -> float:
        """Extract perimeter from embedding."""
        sign = float(embedding[self.PERIMETER_POS])
        log_val = float(embedding[self.PERIMETER_POS + 1])
        return log_decode_value(sign, log_val)

    def get_shape_type(self, embedding: Any) -> ShapeType:
        """Extract shape type from embedding."""
        type_val = int(float(embedding[self.SHAPE_TYPE_POS]) + 0.5)
        return ShapeType(type_val)
