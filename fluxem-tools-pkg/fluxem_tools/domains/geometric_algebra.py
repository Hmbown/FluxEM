"""Geometric algebra domain - Clifford algebra Cl(3,0).

This module provides deterministic geometric algebra computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

# Multivector components in Cl(3,0):
# [scalar, e1, e2, e3, e12, e23, e31, e123]
# where e12 = e1*e2, etc.

def vector_to_multivector(v: List[float]) -> List[float]:
    """Convert 3D vector to multivector representation.

    Args:
        v: [x, y, z] vector

    Returns:
        [0, x, y, z, 0, 0, 0, 0] multivector (vector part only)
    """
    if len(v) != 3:
        raise ValueError("Expected 3D vector")
    return [0.0, v[0], v[1], v[2], 0.0, 0.0, 0.0, 0.0]


def multivector_to_components(mv: List[float]) -> Dict[str, float]:
    """Extract named components from multivector.

    Args:
        mv: 8-component multivector

    Returns:
        Dict with named components
    """
    if len(mv) < 8:
        mv = mv + [0.0] * (8 - len(mv))
    return {
        "scalar": mv[0],
        "e1": mv[1], "e2": mv[2], "e3": mv[3],
        "e12": mv[4], "e23": mv[5], "e31": mv[6],
        "e123": mv[7]
    }


def geometric_product(a: List[float], b: List[float]) -> List[float]:
    """Compute geometric product of two multivectors in Cl(3,0).

    The geometric product ab combines the dot and wedge products.

    Args:
        a: First multivector (8 components)
        b: Second multivector (8 components)

    Returns:
        Product multivector (8 components)
    """
    # Ensure 8 components
    if len(a) == 3:
        a = vector_to_multivector(a)
    if len(b) == 3:
        b = vector_to_multivector(b)

    a = list(a) + [0.0] * (8 - len(a))
    b = list(b) + [0.0] * (8 - len(b))

    # Extract components
    a0, a1, a2, a3, a12, a23, a31, a123 = a
    b0, b1, b2, b3, b12, b23, b31, b123 = b

    # Compute product using Cl(3,0) multiplication table
    # e1*e1 = e2*e2 = e3*e3 = 1
    # e12*e12 = e23*e23 = e31*e31 = -1
    # e123*e123 = -1

    c0 = (a0*b0 + a1*b1 + a2*b2 + a3*b3
          - a12*b12 - a23*b23 - a31*b31 - a123*b123)

    c1 = (a0*b1 + a1*b0 - a2*b12 + a12*b2 + a3*b31 - a31*b3
          - a23*b123 - a123*b23)

    c2 = (a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3
          - a31*b123 - a123*b31)

    c3 = (a0*b3 + a3*b0 - a1*b31 + a31*b1 + a2*b23 - a23*b2
          - a12*b123 - a123*b12)

    c12 = (a0*b12 + a12*b0 + a1*b2 - a2*b1 + a3*b123 + a123*b3
           + a23*b31 - a31*b23)

    c23 = (a0*b23 + a23*b0 + a2*b3 - a3*b2 + a1*b123 + a123*b1
           + a31*b12 - a12*b31)

    c31 = (a0*b31 + a31*b0 + a3*b1 - a1*b3 + a2*b123 + a123*b2
           + a12*b23 - a23*b12)

    c123 = (a0*b123 + a123*b0 + a1*b23 + a23*b1 + a2*b31 + a31*b2
            + a3*b12 + a12*b3)

    return [c0, c1, c2, c3, c12, c23, c31, c123]


def multivector_magnitude(mv: List[float]) -> float:
    """Compute magnitude (norm) of a multivector.

    Args:
        mv: Multivector (3 or 8 components)

    Returns:
        Magnitude as float
    """
    if len(mv) == 3:
        # Simple 3D vector
        return math.sqrt(sum(x**2 for x in mv))

    mv = list(mv) + [0.0] * (8 - len(mv))
    return math.sqrt(sum(x**2 for x in mv))


def dot_product(a: List[float], b: List[float]) -> float:
    """Compute inner (dot) product of two vectors using Clifford algebra.

    For vectors: a·b = (ab + ba)/2 which equals the scalar part.

    Args:
        a: First 3D vector
        b: Second 3D vector

    Returns:
        Scalar dot product
    """
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Expected 3D vectors")
    return sum(ai * bi for ai, bi in zip(a, b))


def wedge_product(a: List[float], b: List[float]) -> List[float]:
    """Compute outer (wedge) product of two vectors, yielding a bivector.

    For vectors: a∧b = (ab - ba)/2

    Args:
        a: First 3D vector [x1, y1, z1]
        b: Second 3D vector [x2, y2, z2]

    Returns:
        Bivector components [e12, e23, e31] (equivalent to cross product)
    """
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Expected 3D vectors")

    # Wedge product components (same as cross product for 3D vectors)
    e12 = a[0] * b[1] - a[1] * b[0]  # e1∧e2
    e23 = a[1] * b[2] - a[2] * b[1]  # e2∧e3
    e31 = a[2] * b[0] - a[0] * b[2]  # e3∧e1

    return [e12, e23, e31]


def rotor_from_axis_angle(axis: List[float], angle: float) -> List[float]:
    """Create a rotor from axis and angle.

    R = cos(θ/2) + sin(θ/2) * (normalized_axis as bivector)

    Args:
        axis: Rotation axis [x, y, z]
        angle: Rotation angle in radians

    Returns:
        Rotor as multivector
    """
    # Normalize axis
    mag = math.sqrt(sum(x**2 for x in axis))
    if mag == 0:
        return [1, 0, 0, 0, 0, 0, 0, 0]

    nx, ny, nz = [x / mag for x in axis]

    half_angle = angle / 2
    c = math.cos(half_angle)
    s = math.sin(half_angle)

    # Bivector representing axis: n1*e23 + n2*e31 + n3*e12
    return [c, 0, 0, 0, s*nz, s*nx, s*ny, 0]


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_vector_3d(args) -> List[float]:
    if isinstance(args, dict):
        v = args.get("vector", args.get("v"))
        if v is not None:
            return [float(x) for x in v]
    if isinstance(args, (list, tuple)) and len(args) == 3:
        return [float(x) for x in args]
    raise ValueError(f"Cannot parse 3D vector: {args}")


def _parse_two_vectors_3d(args) -> Tuple[List[float], List[float]]:
    if isinstance(args, dict):
        a = args.get("a", args.get("v1", args.get("vector1")))
        b = args.get("b", args.get("v2", args.get("vector2")))
        if a is not None and b is not None:
            return [float(x) for x in a], [float(x) for x in b]
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(x) for x in args[0]], [float(x) for x in args[1]]
    raise ValueError(f"Cannot parse two 3D vectors: {args}")


def _parse_multivector(args) -> List[float]:
    if isinstance(args, dict):
        mv = args.get("mv", args.get("multivector", args.get("vector")))
        if mv is not None:
            return [float(x) for x in mv]
    if isinstance(args, (list, tuple)):
        return [float(x) for x in args]
    raise ValueError(f"Cannot parse multivector: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register geometric algebra tools in the registry."""

    registry.register(ToolSpec(
        name="clifford_encode_vector",
        function=lambda args: vector_to_multivector(_parse_vector_3d(args)),
        description="Encodes a 3D vector as Clifford algebra multivector components.",
        parameters={
            "type": "object",
            "properties": {
                "vector": {"type": "array", "items": {"type": "number"}, "description": "3D vector [x, y, z]"}
            },
            "required": ["vector"]
        },
        returns="Multivector [scalar, e1, e2, e3, e12, e23, e31, e123]",
        examples=[
            {"input": {"vector": [1, 2, 3]}, "output": [0, 1, 2, 3, 0, 0, 0, 0]},
        ],
        domain="geometric_algebra",
        tags=["clifford", "vector", "encode"],
    ))

    registry.register(ToolSpec(
        name="clifford_geometric_product",
        function=lambda args: geometric_product(*_parse_two_vectors_3d(args)),
        description="Computes geometric product of two multivectors in Cl(3,0).",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "number"}, "description": "First vector/multivector"},
                "b": {"type": "array", "items": {"type": "number"}, "description": "Second vector/multivector"},
            },
            "required": ["a", "b"]
        },
        returns="Product multivector",
        examples=[
            {"input": {"a": [1, 0, 0], "b": [0, 1, 0]}, "output": [0, 0, 0, 0, 1, 0, 0, 0]},
        ],
        domain="geometric_algebra",
        tags=["clifford", "geometric product"],
    ))

    registry.register(ToolSpec(
        name="clifford_magnitude",
        function=lambda args: multivector_magnitude(_parse_multivector(args)),
        description="Computes magnitude (norm) of a multivector.",
        parameters={
            "type": "object",
            "properties": {
                "vector": {"type": "array", "items": {"type": "number"}, "description": "Multivector or 3D vector"}
            },
            "required": ["vector"]
        },
        returns="Magnitude as float",
        examples=[
            {"input": {"vector": [3, 4, 0]}, "output": 5.0},
        ],
        domain="geometric_algebra",
        tags=["clifford", "magnitude", "norm"],
    ))

    registry.register(ToolSpec(
        name="clifford_dot_product",
        function=lambda args: dot_product(*_parse_two_vectors_3d(args)),
        description="Computes inner (dot) product of two vectors using Clifford algebra.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "number"}, "description": "First 3D vector"},
                "b": {"type": "array", "items": {"type": "number"}, "description": "Second 3D vector"},
            },
            "required": ["a", "b"]
        },
        returns="Scalar dot product",
        examples=[
            {"input": {"a": [1, 0, 0], "b": [1, 0, 0]}, "output": 1.0},
            {"input": {"a": [1, 0, 0], "b": [0, 1, 0]}, "output": 0.0},
        ],
        domain="geometric_algebra",
        tags=["clifford", "dot product", "inner"],
    ))

    registry.register(ToolSpec(
        name="clifford_wedge_product",
        function=lambda args: wedge_product(*_parse_two_vectors_3d(args)),
        description="Computes outer (wedge) product of two vectors, yielding a bivector.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "number"}, "description": "First 3D vector"},
                "b": {"type": "array", "items": {"type": "number"}, "description": "Second 3D vector"},
            },
            "required": ["a", "b"]
        },
        returns="Bivector components [e12, e23, e31]",
        examples=[
            {"input": {"a": [1, 0, 0], "b": [0, 1, 0]}, "output": [1, 0, 0]},
        ],
        domain="geometric_algebra",
        tags=["clifford", "wedge product", "bivector"],
    ))
