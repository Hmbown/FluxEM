"""Control systems domain - state-space, stability analysis.

This module provides deterministic control systems computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def state_update(A: List[List[float]], B: List[List[float]],
                 x: List[float], u: List[float]) -> List[float]:
    """Compute discrete-time state update: x_{t+1} = Ax_t + Bu_t.

    Args:
        A: State transition matrix
        B: Input matrix
        x: Current state vector
        u: Input vector

    Returns:
        Next state vector
    """
    n = len(x)

    # Compute Ax
    ax = [0.0] * n
    for i in range(n):
        for j in range(n):
            ax[i] += A[i][j] * x[j]

    # Compute Bu
    bu = [0.0] * n
    for i in range(n):
        for j in range(len(u)):
            bu[i] += B[i][j] * u[j]

    return [ax[i] + bu[i] for i in range(n)]


def is_stable_2x2(A: List[List[float]]) -> bool:
    """Check stability of 2x2 discrete-time system.

    A system is stable if all eigenvalues have magnitude < 1.

    Args:
        A: 2x2 state transition matrix

    Returns:
        True if stable (all eigenvalues inside unit circle)
    """
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    # Eigenvalues of 2x2 matrix via characteristic polynomial:
    # λ² - (a+d)λ + (ad-bc) = 0
    trace = a + d
    det = a * d - b * c

    discriminant = trace ** 2 - 4 * det

    if discriminant >= 0:
        sqrt_d = math.sqrt(discriminant)
        lambda1 = (trace + sqrt_d) / 2
        lambda2 = (trace - sqrt_d) / 2
        return abs(lambda1) < 1 and abs(lambda2) < 1
    else:
        # Complex eigenvalues
        real = trace / 2
        imag = math.sqrt(-discriminant) / 2
        magnitude = math.sqrt(real ** 2 + imag ** 2)
        return magnitude < 1


def is_controllable_2x2(A: List[List[float]], B: List[List[float]]) -> bool:
    """Check controllability of 2x2 system.

    A system (A, B) is controllable if [B | AB] has full rank.

    Args:
        A: 2x2 state matrix
        B: 2x1 input matrix

    Returns:
        True if controllable
    """
    # B is 2x1
    b1, b2 = B[0][0], B[1][0]

    # AB is 2x1
    ab1 = A[0][0] * b1 + A[0][1] * b2
    ab2 = A[1][0] * b1 + A[1][1] * b2

    # Controllability matrix [B | AB] is 2x2
    det = b1 * ab2 - b2 * ab1

    return abs(det) > 1e-10


def eigenvalues_2x2(A: List[List[float]]) -> List[complex]:
    """Compute eigenvalues of a 2x2 matrix.

    Args:
        A: 2x2 matrix

    Returns:
        List of 2 eigenvalues (may be complex)
    """
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det

    if discriminant >= 0:
        sqrt_d = math.sqrt(discriminant)
        return [(trace + sqrt_d) / 2, (trace - sqrt_d) / 2]
    else:
        real = trace / 2
        imag = math.sqrt(-discriminant) / 2
        return [complex(real, imag), complex(real, -imag)]


def transfer_function_poles(num: List[float], den: List[float]) -> List[complex]:
    """Find poles of a transfer function (roots of denominator).

    For degree 2 only (quadratic denominator).

    Args:
        num: Numerator coefficients [b_n, ..., b_0]
        den: Denominator coefficients [a_n, ..., a_0]

    Returns:
        List of poles
    """
    if len(den) != 3:
        raise ValueError("Only quadratic denominators supported")

    a, b, c = den
    discriminant = b ** 2 - 4 * a * c

    if discriminant >= 0:
        sqrt_d = math.sqrt(discriminant)
        return [(-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)]
    else:
        real = -b / (2 * a)
        imag = math.sqrt(-discriminant) / (2 * a)
        return [complex(real, imag), complex(real, -imag)]


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_state_update(args) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    if isinstance(args, dict):
        A = [[float(x) for x in row] for row in args.get("A")]
        B = [[float(x) for x in row] for row in args.get("B")]
        x = [float(v) for v in args.get("x", args.get("state"))]
        u = [float(v) for v in args.get("u", args.get("input"))]
        return A, B, x, u
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return (
            [[float(x) for x in row] for row in args[0]],
            [[float(x) for x in row] for row in args[1]],
            [float(v) for v in args[2]],
            [float(v) for v in args[3]],
        )
    raise ValueError(f"Cannot parse state update args: {args}")


def _parse_matrix_2x2(args) -> List[List[float]]:
    if isinstance(args, dict):
        A = args.get("A", args.get("matrix"))
        return [[float(x) for x in row] for row in A]
    if isinstance(args, (list, tuple)) and len(args) == 2:
        if isinstance(args[0], (list, tuple)):
            return [[float(x) for x in row] for row in args]
    raise ValueError(f"Cannot parse 2x2 matrix: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register control systems tools in the registry."""

    registry.register(ToolSpec(
        name="control_state_update",
        function=lambda args: state_update(*_parse_state_update(args)),
        description="Computes discrete-time state update: x_{t+1} = Ax_t + Bu_t.",
        parameters={
            "type": "object",
            "properties": {
                "A": {"type": "array", "description": "State transition matrix"},
                "B": {"type": "array", "description": "Input matrix"},
                "x": {"type": "array", "items": {"type": "number"}, "description": "Current state"},
                "u": {"type": "array", "items": {"type": "number"}, "description": "Input vector"},
            },
            "required": ["A", "B", "x", "u"]
        },
        returns="Next state vector",
        examples=[
            {"input": {"A": [[1, 0], [0, 1]], "B": [[1, 0], [0, 1]], "x": [1, 2], "u": [0, 1]}, "output": [1.0, 3.0]},
        ],
        domain="control_systems",
        tags=["state", "update", "discrete"],
    ))

    registry.register(ToolSpec(
        name="control_is_stable_2x2",
        function=lambda args: is_stable_2x2(_parse_matrix_2x2(args)),
        description="Checks stability of 2x2 discrete-time system (eigenvalues in unit circle).",
        parameters={
            "type": "object",
            "properties": {
                "A": {"type": "array", "description": "2x2 state transition matrix"}
            },
            "required": ["A"]
        },
        returns="Boolean: True if stable",
        examples=[
            {"input": {"A": [[0.5, 0], [0, 0.5]]}, "output": True},
            {"input": {"A": [[2, 0], [0, 0.5]]}, "output": False},
        ],
        domain="control_systems",
        tags=["stability", "eigenvalue", "discrete"],
    ))
