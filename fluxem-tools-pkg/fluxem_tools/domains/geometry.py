"""Geometry domain - points, distances, transformations.

This module provides deterministic geometry computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    """Compute Euclidean distance between two points.

    Args:
        p1: First point as coordinate list
        p2: Second point as coordinate list

    Returns:
        Distance as float
    """
    if len(p1) != len(p2):
        raise ValueError(f"Point dimensions must match: {len(p1)} vs {len(p2)}")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def midpoint(p1: List[float], p2: List[float]) -> List[float]:
    """Compute midpoint between two points.

    Args:
        p1: First point as coordinate list
        p2: Second point as coordinate list

    Returns:
        Midpoint coordinates
    """
    if len(p1) != len(p2):
        raise ValueError(f"Point dimensions must match: {len(p1)} vs {len(p2)}")
    return [(a + b) / 2 for a, b in zip(p1, p2)]


def rotate_2d(point: List[float], angle: float, center: List[float] = None) -> List[float]:
    """Rotate a 2D point around a center (default origin) by angle in radians.

    Args:
        point: Point as [x, y]
        angle: Rotation angle in radians (counterclockwise positive)
        center: Center of rotation, default [0, 0]

    Returns:
        Rotated point as [x, y]
    """
    if center is None:
        center = [0.0, 0.0]

    x, y = point[0] - center[0], point[1] - center[1]
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a

    return [x_new + center[0], y_new + center[1]]


def triangle_area(p1: List[float], p2: List[float], p3: List[float]) -> float:
    """Compute area of a triangle from three vertices using the shoelace formula.

    Args:
        p1, p2, p3: Triangle vertices as [x, y] lists

    Returns:
        Area as float
    """
    return abs((p1[0] * (p2[1] - p3[1]) +
                p2[0] * (p3[1] - p1[1]) +
                p3[0] * (p1[1] - p2[1])) / 2)


def polygon_area(vertices: List[List[float]]) -> float:
    """Compute area of a polygon using the shoelace formula.

    Args:
        vertices: List of [x, y] vertices in order

    Returns:
        Area as float
    """
    n = len(vertices)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return abs(area) / 2


def circle_area(radius: float) -> float:
    """Compute area of a circle."""
    return math.pi * radius ** 2


def circle_circumference(radius: float) -> float:
    """Compute circumference of a circle."""
    return 2 * math.pi * radius


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_two_points(args) -> Tuple[List[float], List[float]]:
    if isinstance(args, dict):
        p1 = args.get("p1", args.get("point1", args.get("a")))
        p2 = args.get("p2", args.get("point2", args.get("b")))
        if p1 is not None and p2 is not None:
            return [float(x) for x in p1], [float(x) for x in p2]
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(x) for x in args[0]], [float(x) for x in args[1]]
    raise ValueError(f"Cannot parse two points: {args}")


def _parse_rotate_args(args) -> Tuple[List[float], float]:
    if isinstance(args, dict):
        point = args.get("point", args.get("p"))
        angle = args.get("angle", args.get("radians", args.get("theta")))
        return [float(x) for x in point], float(angle)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(x) for x in args[0]], float(args[1])
    raise ValueError(f"Cannot parse rotate args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register geometry tools in the registry."""

    registry.register(ToolSpec(
        name="geometry_distance",
        function=lambda args: euclidean_distance(*_parse_two_points(args)),
        description="Computes Euclidean distance between two points (any dimension).",
        parameters={
            "type": "object",
            "properties": {
                "p1": {"type": "array", "items": {"type": "number"}, "description": "First point"},
                "p2": {"type": "array", "items": {"type": "number"}, "description": "Second point"},
            },
            "required": ["p1", "p2"]
        },
        returns="Distance as float",
        examples=[
            {"input": {"p1": [0, 0], "p2": [3, 4]}, "output": 5.0},
            {"input": {"p1": [0, 0, 0], "p2": [1, 2, 2]}, "output": 3.0},
        ],
        domain="geometry",
        tags=["distance", "euclidean", "points"],
    ))

    registry.register(ToolSpec(
        name="geometry_midpoint",
        function=lambda args: midpoint(*_parse_two_points(args)),
        description="Computes midpoint between two points.",
        parameters={
            "type": "object",
            "properties": {
                "p1": {"type": "array", "items": {"type": "number"}, "description": "First point"},
                "p2": {"type": "array", "items": {"type": "number"}, "description": "Second point"},
            },
            "required": ["p1", "p2"]
        },
        returns="Midpoint as coordinate list",
        examples=[
            {"input": {"p1": [0, 0], "p2": [4, 6]}, "output": [2.0, 3.0]},
        ],
        domain="geometry",
        tags=["midpoint", "center", "points"],
    ))

    registry.register(ToolSpec(
        name="geometry_rotate",
        function=lambda args: rotate_2d(*_parse_rotate_args(args)),
        description="Rotates a 2D point around the origin by an angle in radians.",
        parameters={
            "type": "object",
            "properties": {
                "point": {"type": "array", "items": {"type": "number"}, "description": "2D point [x, y]"},
                "angle": {"type": "number", "description": "Rotation angle in radians"},
            },
            "required": ["point", "angle"]
        },
        returns="Rotated point as [x, y]",
        examples=[
            {"input": {"point": [1, 0], "angle": 1.5708}, "output": [0.0, 1.0]},
        ],
        domain="geometry",
        tags=["rotate", "transformation", "2d"],
    ))
