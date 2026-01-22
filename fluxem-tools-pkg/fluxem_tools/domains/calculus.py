"""Calculus domain - polynomial operations, derivatives, integrals.

This module provides deterministic calculus computations for polynomials.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions (coefficients in descending order: [a_n, ..., a_1, a_0])
# =============================================================================

def poly_derivative(coeffs: List[float]) -> List[float]:
    """Compute derivative of a polynomial.

    Args:
        coeffs: Coefficients [a_n, a_{n-1}, ..., a_1, a_0] (highest degree first)

    Returns:
        Derivative coefficients (highest degree first)
    """
    if len(coeffs) <= 1:
        return [0.0]

    n = len(coeffs) - 1  # degree
    result = []
    for i, c in enumerate(coeffs[:-1]):
        power = n - i
        result.append(c * power)

    return result if result else [0.0]


def poly_integral(coeffs: List[float], constant: float = 0.0) -> List[float]:
    """Compute indefinite integral of a polynomial.

    Args:
        coeffs: Coefficients [a_n, ..., a_0] (highest degree first)
        constant: Integration constant (default 0)

    Returns:
        Integral coefficients (highest degree first)
    """
    if not coeffs:
        return [constant]

    n = len(coeffs) - 1  # current degree
    result = []
    for i, c in enumerate(coeffs):
        power = n - i
        result.append(c / (power + 1))

    result.append(constant)
    return result


def poly_evaluate(coeffs: List[float], x: float) -> float:
    """Evaluate a polynomial at x using Horner's method.

    Args:
        coeffs: Coefficients [a_n, ..., a_0] (highest degree first)
        x: Point to evaluate at

    Returns:
        Polynomial value at x
    """
    if not coeffs:
        return 0.0

    result = coeffs[0]
    for c in coeffs[1:]:
        result = result * x + c
    return result


def poly_add(p1: List[float], p2: List[float]) -> List[float]:
    """Add two polynomials."""
    # Pad shorter polynomial with zeros
    n1, n2 = len(p1), len(p2)
    if n1 < n2:
        p1 = [0.0] * (n2 - n1) + p1
    elif n2 < n1:
        p2 = [0.0] * (n1 - n2) + p2

    result = [a + b for a, b in zip(p1, p2)]

    # Remove leading zeros
    while len(result) > 1 and result[0] == 0:
        result.pop(0)

    return result


def poly_multiply(p1: List[float], p2: List[float]) -> List[float]:
    """Multiply two polynomials."""
    if not p1 or not p2:
        return [0.0]

    n = len(p1) + len(p2) - 1
    result = [0.0] * n

    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            result[i + j] += a * b

    return result


def poly_roots_quadratic(coeffs: List[float]) -> List[complex]:
    """Find roots of a quadratic polynomial ax² + bx + c.

    Args:
        coeffs: [a, b, c] (highest degree first)

    Returns:
        List of roots (may be complex)
    """
    if len(coeffs) != 3:
        raise ValueError("Expected 3 coefficients for quadratic")

    a, b, c = coeffs
    if a == 0:
        if b == 0:
            return []
        return [-c / b]

    discriminant = b ** 2 - 4 * a * c

    if discriminant >= 0:
        sqrt_d = discriminant ** 0.5
        return [(-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)]
    else:
        sqrt_d = (-discriminant) ** 0.5
        real = -b / (2 * a)
        imag = sqrt_d / (2 * a)
        return [complex(real, imag), complex(real, -imag)]


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_coeffs(args) -> List[float]:
    """Parse polynomial coefficients (expects highest degree first)."""
    if isinstance(args, dict):
        coeffs = args.get("coeffs", args.get("coefficients", args.get("polynomial")))
        if coeffs is not None:
            return [float(c) for c in coeffs]
        return [float(c) for c in list(args.values())[0]]
    if isinstance(args, (list, tuple)):
        return [float(c) for c in args]
    raise ValueError(f"Cannot parse coefficients: {args}")


def _parse_eval_args(args) -> Tuple[List[float], float]:
    if isinstance(args, dict):
        coeffs = args.get("coeffs", args.get("coefficients", args.get("polynomial")))
        x = args.get("x", args.get("value", args.get("point")))
        return [float(c) for c in coeffs], float(x)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(c) for c in args[0]], float(args[1])
    raise ValueError(f"Cannot parse eval args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register calculus tools in the registry."""

    registry.register(ToolSpec(
        name="calculus_derivative",
        function=lambda args: poly_derivative(_parse_coeffs(args)),
        description="Computes derivative coefficients of a polynomial (highest degree first).",
        parameters={
            "type": "object",
            "properties": {
                "coeffs": {"type": "array", "items": {"type": "number"}, "description": "Coefficients [a_n, ..., a_0]"}
            },
            "required": ["coeffs"]
        },
        returns="Derivative coefficients",
        examples=[
            {"input": {"coeffs": [1, 2, 3]}, "output": [2, 2]},  # x² + 2x + 3 → 2x + 2
            {"input": {"coeffs": [3, 2, 1]}, "output": [6, 2]},  # 3x² + 2x + 1 → 6x + 2
        ],
        domain="calculus",
        tags=["derivative", "polynomial", "differentiation"],
    ))

    registry.register(ToolSpec(
        name="calculus_integral",
        function=lambda args: poly_integral(_parse_coeffs(args)),
        description="Computes integral coefficients of a polynomial (constant = 0, highest degree first).",
        parameters={
            "type": "object",
            "properties": {
                "coeffs": {"type": "array", "items": {"type": "number"}, "description": "Coefficients [a_n, ..., a_0]"}
            },
            "required": ["coeffs"]
        },
        returns="Integral coefficients",
        examples=[
            {"input": {"coeffs": [2, 6]}, "output": [1.0, 6.0, 0.0]},  # 2x + 6 → x² + 6x + C
        ],
        domain="calculus",
        tags=["integral", "polynomial", "antiderivative"],
    ))

    registry.register(ToolSpec(
        name="calculus_evaluate",
        function=lambda args: poly_evaluate(*_parse_eval_args(args)),
        description="Evaluates a polynomial at x (coefficients highest degree first).",
        parameters={
            "type": "object",
            "properties": {
                "coeffs": {"type": "array", "items": {"type": "number"}, "description": "Coefficients [a_n, ..., a_0]"},
                "x": {"type": "number", "description": "Point to evaluate at"},
            },
            "required": ["coeffs", "x"]
        },
        returns="Polynomial value as float",
        examples=[
            {"input": {"coeffs": [1, 2, 3], "x": 2}, "output": 11.0},  # 1(4) + 2(2) + 3 = 11
        ],
        domain="calculus",
        tags=["evaluate", "polynomial", "horner"],
    ))
