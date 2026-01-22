"""Advanced mathematics domain - vectors, matrices, complex numbers.

This module provides deterministic advanced math computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Vector Operations
# =============================================================================

def vector_magnitude(v: List[float]) -> float:
    """Compute the magnitude (L2 norm) of a vector."""
    return math.sqrt(sum(x**2 for x in v))


def vector_normalize(v: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    mag = vector_magnitude(v)
    if mag == 0:
        return [0.0] * len(v)
    return [x / mag for x in v]


def vector_dot(a: List[float], b: List[float]) -> float:
    """Compute dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def vector_cross(a: List[float], b: List[float]) -> List[float]:
    """Compute cross product of two 3D vectors."""
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Cross product requires 3D vectors")
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


# =============================================================================
# Matrix Operations
# =============================================================================

def matrix_determinant(m: List[List[float]]) -> float:
    """Compute determinant of a square matrix (up to 4x4)."""
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if n == 3:
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if n == 4:
        # Laplace expansion along first row
        det = 0
        for j in range(4):
            minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
            det += ((-1) ** j) * m[0][j] * matrix_determinant(minor)
        return det
    raise ValueError(f"Matrix too large: {n}x{n}")


def matrix_transpose(m: List[List[float]]) -> List[List[float]]:
    """Compute transpose of a matrix."""
    if not m:
        return []
    rows = len(m)
    cols = len(m[0])
    return [[m[i][j] for i in range(rows)] for j in range(cols)]


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices."""
    if not a or not b:
        return []
    m, n = len(a), len(a[0])
    p = len(b[0])
    if n != len(b):
        raise ValueError(f"Matrix dimensions incompatible: {m}x{n} and {len(b)}x{p}")

    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_vector(args) -> List[float]:
    if isinstance(args, dict):
        if "vector" in args:
            return [float(x) for x in args["vector"]]
        if "v" in args:
            return [float(x) for x in args["v"]]
        # Assume first value is the vector
        for v in args.values():
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
    if isinstance(args, (list, tuple)):
        return [float(x) for x in args]
    raise ValueError(f"Cannot parse vector: {args}")


def _parse_two_vectors(args) -> Tuple[List[float], List[float]]:
    if isinstance(args, dict):
        a = args.get("a", args.get("v1", args.get("vector1")))
        b = args.get("b", args.get("v2", args.get("vector2")))
        if a is not None and b is not None:
            return [float(x) for x in a], [float(x) for x in b]
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(x) for x in args[0]], [float(x) for x in args[1]]
    raise ValueError(f"Cannot parse two vectors: {args}")


def _parse_matrix(args) -> List[List[float]]:
    if isinstance(args, dict):
        if "matrix" in args:
            return [[float(x) for x in row] for row in args["matrix"]]
        if "m" in args:
            return [[float(x) for x in row] for row in args["m"]]
    if isinstance(args, (list, tuple)) and args and isinstance(args[0], (list, tuple)):
        return [[float(x) for x in row] for row in args]
    raise ValueError(f"Cannot parse matrix: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register advanced math tools in the registry."""

    registry.register(ToolSpec(
        name="math_vector",
        function=lambda args: vector_magnitude(_parse_vector(args)),
        description="Computes the magnitude (L2 norm) of a vector.",
        parameters={
            "type": "object",
            "properties": {
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Vector as list of numbers"
                }
            },
            "required": ["vector"]
        },
        returns="Magnitude as float",
        examples=[
            {"input": {"vector": [3, 4]}, "output": 5.0},
            {"input": {"vector": [1, 2, 2]}, "output": 3.0},
        ],
        domain="math",
        tags=["vector", "magnitude", "norm"],
    ))

    registry.register(ToolSpec(
        name="math_dot",
        function=lambda args: vector_dot(*_parse_two_vectors(args)),
        description="Computes dot product of two vectors.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "number"}, "description": "First vector"},
                "b": {"type": "array", "items": {"type": "number"}, "description": "Second vector"},
            },
            "required": ["a", "b"]
        },
        returns="Dot product as float",
        examples=[
            {"input": {"a": [1, 2], "b": [3, 4]}, "output": 11.0},
            {"input": {"a": [1, 0, 0], "b": [0, 1, 0]}, "output": 0.0},
        ],
        domain="math",
        tags=["vector", "dot product", "inner product"],
    ))

    registry.register(ToolSpec(
        name="math_determinant",
        function=lambda args: matrix_determinant(_parse_matrix(args)),
        description="Computes determinant of a square matrix (up to 4x4).",
        parameters={
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Square matrix as 2D array"
                }
            },
            "required": ["matrix"]
        },
        returns="Determinant as float",
        examples=[
            {"input": {"matrix": [[1, 2], [3, 4]]}, "output": -2.0},
            {"input": {"matrix": [[1, 0], [0, 1]]}, "output": 1.0},
        ],
        domain="math",
        tags=["matrix", "determinant", "linear algebra"],
    ))

    registry.register(ToolSpec(
        name="math_normalize",
        function=lambda args: vector_normalize(_parse_vector(args)),
        description="Normalizes a vector to unit length.",
        parameters={
            "type": "object",
            "properties": {
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Vector to normalize"
                }
            },
            "required": ["vector"]
        },
        returns="Normalized vector as list",
        examples=[
            {"input": {"vector": [3, 4]}, "output": [0.6, 0.8]},
            {"input": {"vector": [1, 0, 0]}, "output": [1.0, 0.0, 0.0]},
        ],
        domain="math",
        tags=["vector", "normalize", "unit"],
    ))

    registry.register(ToolSpec(
        name="math_cross_product",
        function=lambda args: vector_cross(*_parse_two_vectors(args)),
        description="Computes the cross product of two 3D vectors.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "number"}, "description": "First 3D vector"},
                "b": {"type": "array", "items": {"type": "number"}, "description": "Second 3D vector"},
            },
            "required": ["a", "b"]
        },
        returns="Cross product as 3D vector",
        examples=[
            {"input": {"a": [1, 0, 0], "b": [0, 1, 0]}, "output": [0, 0, 1]},
            {"input": {"a": [1, 2, 3], "b": [4, 5, 6]}, "output": [-3, 6, -3]},
        ],
        domain="math",
        tags=["vector", "cross product", "3d"],
    ))

    registry.register(ToolSpec(
        name="math_matrix_transpose",
        function=lambda args: matrix_transpose(_parse_matrix(args)),
        description="Computes the transpose of a matrix.",
        parameters={
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Matrix as 2D array"
                }
            },
            "required": ["matrix"]
        },
        returns="Transposed matrix",
        examples=[
            {"input": {"matrix": [[1, 2], [3, 4]]}, "output": [[1, 3], [2, 4]]},
            {"input": {"matrix": [[1, 2, 3]]}, "output": [[1], [2], [3]]},
        ],
        domain="math",
        tags=["matrix", "transpose", "linear algebra"],
    ))
