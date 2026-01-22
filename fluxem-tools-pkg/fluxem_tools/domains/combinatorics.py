"""Combinatorics domain - factorial, permutations, combinations.

This module provides deterministic combinatorics computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def factorial(n: int) -> int:
    """Compute factorial n! for n >= 0."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    return math.factorial(n)


def ncr(n: int, k: int) -> int:
    """Compute combinations C(n, k) = n! / (k!(n-k)!).

    Args:
        n: Total items
        k: Items to choose

    Returns:
        Number of combinations
    """
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def npr(n: int, k: int) -> int:
    """Compute permutations P(n, k) = n! / (n-k)!.

    Args:
        n: Total items
        k: Items to arrange

    Returns:
        Number of permutations
    """
    if k < 0 or k > n:
        return 0
    return math.perm(n, k)


def multiset_permutations(counts: List[int]) -> int:
    """Compute multinomial coefficient: n! / (n1! * n2! * ... * nk!).

    Args:
        counts: List of item counts [n1, n2, ..., nk]

    Returns:
        Number of permutations
    """
    n = sum(counts)
    result = factorial(n)
    for c in counts:
        result //= factorial(c)
    return result


def multiset_combinations(n: int, k: int) -> int:
    """Compute combinations with repetition: C(n+k-1, k).

    Args:
        n: Number of types
        k: Number to choose

    Returns:
        Number of combinations with repetition
    """
    return ncr(n + k - 1, k)


def catalan_number(n: int) -> int:
    """Compute the n-th Catalan number: C_n = C(2n, n) / (n+1)."""
    if n < 0:
        raise ValueError("Catalan number not defined for negative n")
    return ncr(2 * n, n) // (n + 1)


def derangements(n: int) -> int:
    """Compute number of derangements (permutations with no fixed points).

    Uses the formula: !n = n! * Σ((-1)^k / k!) for k=0 to n
    """
    if n < 0:
        raise ValueError("Derangements not defined for negative n")
    if n == 0:
        return 1
    if n == 1:
        return 0

    result = 0
    fact_n = factorial(n)
    for k in range(n + 1):
        result += ((-1) ** k) * fact_n // factorial(k)
    return result


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_single_int(args) -> int:
    if isinstance(args, dict):
        n = args.get("n", args.get("number", list(args.values())[0]))
        return int(n)
    if isinstance(args, (list, tuple)):
        return int(args[0])
    return int(args)


def _parse_int_pair(args) -> Tuple[int, int]:
    if isinstance(args, dict):
        n = int(args.get("n", args.get("total")))
        k = int(args.get("k", args.get("r", args.get("choose"))))
        return n, k
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return int(args[0]), int(args[1])
    raise ValueError(f"Cannot parse int pair: {args}")


def _parse_multiset(args):
    if isinstance(args, dict):
        if "counts" in args:
            return multiset_permutations([int(x) for x in args["counts"]])
        if "n" in args and "k" in args:
            return multiset_combinations(int(args["n"]), int(args["k"]))
    if isinstance(args, (list, tuple)):
        if len(args) == 2 and not isinstance(args[0], (list, tuple)):
            return multiset_combinations(int(args[0]), int(args[1]))
        return multiset_permutations([int(x) for x in args])
    raise ValueError(f"Cannot parse multiset args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register combinatorics tools in the registry."""

    registry.register(ToolSpec(
        name="combinatorics_factorial",
        function=lambda args: factorial(_parse_single_int(args)),
        description="Computes factorial n! for n >= 0.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Non-negative integer"}
            },
            "required": ["n"]
        },
        returns="Factorial as integer",
        examples=[
            {"input": {"n": 5}, "output": 120},
            {"input": {"n": 10}, "output": 3628800},
        ],
        domain="combinatorics",
        tags=["factorial", "permutation"],
    ))

    registry.register(ToolSpec(
        name="combinatorics_ncr",
        function=lambda args: ncr(*_parse_int_pair(args)),
        description="Computes combinations C(n, k) = n! / (k!(n-k)!).",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Total items"},
                "k": {"type": "integer", "description": "Items to choose"},
            },
            "required": ["n", "k"]
        },
        returns="Number of combinations as integer",
        examples=[
            {"input": {"n": 5, "k": 2}, "output": 10},
            {"input": {"n": 10, "k": 3}, "output": 120},
        ],
        domain="combinatorics",
        tags=["combination", "choose", "binomial"],
    ))

    registry.register(ToolSpec(
        name="combinatorics_npr",
        function=lambda args: npr(*_parse_int_pair(args)),
        description="Computes permutations P(n, k) = n! / (n-k)!.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Total items"},
                "k": {"type": "integer", "description": "Items to arrange"},
            },
            "required": ["n", "k"]
        },
        returns="Number of permutations as integer",
        examples=[
            {"input": {"n": 5, "k": 2}, "output": 20},
            {"input": {"n": 10, "k": 3}, "output": 720},
        ],
        domain="combinatorics",
        tags=["permutation", "arrangement"],
    ))

    registry.register(ToolSpec(
        name="combinatorics_multiset",
        function=_parse_multiset,
        description="Computes combinations with repetition C(n+k-1, k) or multinomial permutations.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of types (for combinations with repetition)"},
                "k": {"type": "integer", "description": "Number to choose"},
            },
            "required": ["n", "k"]
        },
        returns="Count as integer",
        examples=[
            {"input": {"n": 3, "k": 2}, "output": 6},
        ],
        domain="combinatorics",
        tags=["multiset", "repetition", "combination"],
    ))
