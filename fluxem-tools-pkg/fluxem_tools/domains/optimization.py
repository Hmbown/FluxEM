"""Optimization domain - least squares, gradient descent, projections.

This module provides deterministic optimization computations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def least_squares_2x2(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve 2x2 linear system Ax = b using Cramer's rule.

    Args:
        A: 2x2 coefficient matrix [[a11, a12], [a21, a22]]
        b: Right-hand side vector [b1, b2]

    Returns:
        Solution vector [x1, x2]

    Raises:
        ValueError: If matrix is singular
    """
    a11, a12 = A[0][0], A[0][1]
    a21, a22 = A[1][0], A[1][1]
    b1, b2 = b[0], b[1]

    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-15:
        raise ValueError("Matrix is singular")

    x1 = (b1 * a22 - a12 * b2) / det
    x2 = (a11 * b2 - b1 * a21) / det

    return [x1, x2]


def gradient_step(x: List[float], grad: List[float], lr: float) -> List[float]:
    """Apply one gradient descent step: x_new = x - lr * grad.

    Args:
        x: Current position
        grad: Gradient vector
        lr: Learning rate

    Returns:
        Updated position
    """
    return [xi - lr * gi for xi, gi in zip(x, grad)]


def project_box(x: List[float], lower: float, upper: float) -> List[float]:
    """Project vector into box constraints [lower, upper].

    Args:
        x: Vector to project
        lower: Lower bound
        upper: Upper bound

    Returns:
        Projected vector
    """
    return [max(lower, min(upper, xi)) for xi in x]


def project_simplex(x: List[float]) -> List[float]:
    """Project vector onto probability simplex (sum=1, all >= 0).

    Uses the algorithm from "Efficient Projections onto the l1-Ball".

    Args:
        x: Vector to project

    Returns:
        Projected vector on simplex
    """
    n = len(x)
    sorted_x = sorted(x, reverse=True)

    cumsum = 0.0
    rho = 0
    for j in range(n):
        cumsum += sorted_x[j]
        if sorted_x[j] + (1 - cumsum) / (j + 1) > 0:
            rho = j + 1

    theta = (cumsum - 1) / rho if rho > 0 else 0

    return [max(xi - theta, 0) for xi in x]


def line_search_backtracking(f, grad_f, x: List[float], direction: List[float],
                              alpha: float = 1.0, beta: float = 0.5, c: float = 1e-4,
                              max_iter: int = 20) -> float:
    """Backtracking line search to find step size.

    Args:
        f: Objective function
        grad_f: Gradient function
        x: Current point
        direction: Search direction
        alpha: Initial step size
        beta: Reduction factor (0 < beta < 1)
        c: Sufficient decrease constant
        max_iter: Maximum iterations

    Returns:
        Step size satisfying Armijo condition
    """
    fx = f(x)
    grad = grad_f(x)
    descent = sum(g * d for g, d in zip(grad, direction))

    for _ in range(max_iter):
        x_new = [xi + alpha * di for xi, di in zip(x, direction)]
        if f(x_new) <= fx + c * alpha * descent:
            return alpha
        alpha *= beta

    return alpha


def golden_section_search(f, a: float, b: float, tol: float = 1e-6) -> float:
    """Find minimum of unimodal function using golden section search.

    Args:
        f: Unimodal function
        a: Left bound
        b: Right bound
        tol: Tolerance

    Returns:
        Approximate minimizer
    """
    phi = (1 + 5 ** 0.5) / 2  # Golden ratio

    c = b - (b - a) / phi
    d = a + (b - a) / phi

    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a) / phi
        d = a + (b - a) / phi

    return (a + b) / 2


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_least_squares(args) -> Tuple[List[List[float]], List[float]]:
    if isinstance(args, dict):
        A = [[float(x) for x in row] for row in args.get("A", args.get("matrix"))]
        b = [float(x) for x in args.get("b", args.get("vector"))]
        return A, b
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return [[float(x) for x in row] for row in args[0]], [float(x) for x in args[1]]
    raise ValueError(f"Cannot parse least squares args: {args}")


def _parse_gradient_step(args) -> Tuple[List[float], List[float], float]:
    if isinstance(args, dict):
        x = [float(v) for v in args.get("x", args.get("position"))]
        grad = [float(v) for v in args.get("grad", args.get("gradient"))]
        lr = float(args.get("lr", args.get("learning_rate", args.get("step_size"))))
        return x, grad, lr
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return [float(v) for v in args[0]], [float(v) for v in args[1]], float(args[2])
    raise ValueError(f"Cannot parse gradient step args: {args}")


def _parse_project_box(args) -> Tuple[List[float], float, float]:
    if isinstance(args, dict):
        x = [float(v) for v in args.get("x", args.get("vector"))]
        lower = float(args.get("lower", args.get("min", -float('inf'))))
        upper = float(args.get("upper", args.get("max", float('inf'))))
        return x, lower, upper
    raise ValueError(f"Cannot parse project box args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register optimization tools in the registry."""

    registry.register(ToolSpec(
        name="optimization_least_squares_2x2",
        function=lambda args: least_squares_2x2(*_parse_least_squares(args)),
        description="Solves a 2x2 linear system Ax = b.",
        parameters={
            "type": "object",
            "properties": {
                "A": {"type": "array", "description": "2x2 coefficient matrix"},
                "b": {"type": "array", "description": "Right-hand side vector"},
            },
            "required": ["A", "b"]
        },
        returns="Solution vector [x1, x2]",
        examples=[
            {"input": {"A": [[1, 2], [3, 4]], "b": [5, 6]}, "output": [-4.0, 4.5]},
        ],
        domain="optimization",
        tags=["linear", "solve", "2x2"],
    ))

    registry.register(ToolSpec(
        name="optimization_gradient_step",
        function=lambda args: gradient_step(*_parse_gradient_step(args)),
        description="Applies one gradient descent step: x_new = x - lr * grad.",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "array", "items": {"type": "number"}, "description": "Current position"},
                "grad": {"type": "array", "items": {"type": "number"}, "description": "Gradient vector"},
                "lr": {"type": "number", "description": "Learning rate"},
            },
            "required": ["x", "grad", "lr"]
        },
        returns="Updated position vector",
        examples=[
            {"input": {"x": [1, 2], "grad": [0.1, 0.2], "lr": 0.5}, "output": [0.95, 1.9]},
        ],
        domain="optimization",
        tags=["gradient", "descent", "step"],
    ))

    registry.register(ToolSpec(
        name="optimization_project_box",
        function=lambda args: project_box(*_parse_project_box(args)),
        description="Projects a vector into box constraints [lower, upper].",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "array", "items": {"type": "number"}, "description": "Vector to project"},
                "lower": {"type": "number", "description": "Lower bound"},
                "upper": {"type": "number", "description": "Upper bound"},
            },
            "required": ["x", "lower", "upper"]
        },
        returns="Projected vector",
        examples=[
            {"input": {"x": [2, -1, 0], "lower": -0.5, "upper": 0.5}, "output": [0.5, -0.5, 0.0]},
        ],
        domain="optimization",
        tags=["projection", "constraint", "box"],
    ))
