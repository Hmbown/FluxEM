"""Arithmetic domain - basic computation tools.

This module provides deterministic arithmetic evaluation.
"""

import ast
import math
import operator
import re
from typing import Any, Dict, Union

from ..registry import ToolSpec, ToolRegistry


def compute_arithmetic(args) -> float:
    """Compute arithmetic expressions with safe evaluation.

    Supports: +, -, *, /, **, %, pi, e

    Args:
        args: Either a string expression or dict with "expr" key

    Returns:
        Computed result as float

    Examples:
        >>> compute_arithmetic("2 + 3")
        5.0
        >>> compute_arithmetic({"expr": "2 ** 10"})
        1024.0
        >>> compute_arithmetic("pi * 2")
        6.283185307179586
    """
    # Handle dict input with "expr" key
    if isinstance(args, dict):
        expr = args.get("expr", args.get("expression"))
        if expr is None:
            raise ValueError("Required: 'expr' key with arithmetic expression")
    else:
        expr = args

    if not isinstance(expr, str):
        raise ValueError(f"Expression must be a string, got: {type(expr)}")

    expr_clean = expr.strip()
    expr_clean = expr_clean.replace("^", "**")
    expr_clean = expr_clean.replace("×", "*")
    expr_clean = re.sub(r"\bpi\b", "pi", expr_clean, flags=re.IGNORECASE)
    expr_clean = re.sub(r"\be\b", "e", expr_clean, flags=re.IGNORECASE)

    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
            return bin_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
        raise ValueError(f"Unsupported expression element: {node}")

    tree = ast.parse(expr_clean, mode="eval")
    return _eval(tree)


def register_tools(registry: ToolRegistry) -> None:
    """Register arithmetic tools in the registry."""

    registry.register(ToolSpec(
        name="arithmetic",
        function=compute_arithmetic,
        description="Evaluates arithmetic expressions with 100% accuracy. Supports +, -, *, /, **, %, pi, e.",
        parameters={
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate (e.g., '2 + 3 * 4', '2**10', 'pi * 2')"
                }
            },
            "required": ["expr"]
        },
        returns="Numeric result as float",
        examples=[
            {"input": {"expr": "2 + 3"}, "output": 5.0},
            {"input": {"expr": "2 ** 10"}, "output": 1024.0},
            {"input": {"expr": "pi * 2"}, "output": 6.283185307179586},
            {"input": {"expr": "54 * 44"}, "output": 2376.0},
        ],
        domain="arithmetic",
        tags=["math", "calculation", "expression", "evaluate"],
    ))
