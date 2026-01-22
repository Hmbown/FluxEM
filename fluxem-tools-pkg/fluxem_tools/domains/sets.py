"""Sets domain - union, intersection, complement, subset.

This module provides deterministic set operations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def set_union(a: List, b: List) -> List:
    """Compute union of two sets (elements in either set)."""
    return sorted(set(a) | set(b), key=lambda x: (type(x).__name__, x))


def set_intersection(a: List, b: List) -> List:
    """Compute intersection of two sets (elements in both sets)."""
    return sorted(set(a) & set(b), key=lambda x: (type(x).__name__, x))


def set_difference(a: List, b: List) -> List:
    """Compute difference A - B (elements in A but not in B)."""
    return sorted(set(a) - set(b), key=lambda x: (type(x).__name__, x))


def set_symmetric_difference(a: List, b: List) -> List:
    """Compute symmetric difference (elements in exactly one set)."""
    return sorted(set(a) ^ set(b), key=lambda x: (type(x).__name__, x))


def is_subset(a: List, b: List) -> bool:
    """Check if A is a subset of B."""
    return set(a) <= set(b)


def is_superset(a: List, b: List) -> bool:
    """Check if A is a superset of B."""
    return set(a) >= set(b)


def set_complement(a: List, universe: List) -> List:
    """Compute complement of A relative to universe."""
    return sorted(set(universe) - set(a), key=lambda x: (type(x).__name__, x))


def cardinality(a: List) -> int:
    """Return the number of unique elements in a set."""
    return len(set(a))


def power_set_size(n: int) -> int:
    """Return the size of the power set (2^n)."""
    return 2 ** n


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_two_sets(args) -> Tuple[List, List]:
    if isinstance(args, dict):
        a = args.get("a", args.get("set1", args.get("first")))
        b = args.get("b", args.get("set2", args.get("second")))
        if a is not None and b is not None:
            return list(a), list(b)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return list(args[0]), list(args[1])
    raise ValueError(f"Cannot parse two sets: {args}")


def _parse_set_operation(args, operation: str):
    a, b = _parse_two_sets(args)
    ops = {
        "union": set_union,
        "intersection": set_intersection,
        "difference": set_difference,
        "symmetric_difference": set_symmetric_difference,
        "subset": is_subset,
        "superset": is_superset,
        "complement": set_complement,
    }
    return ops[operation](a, b)


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register set operation tools in the registry."""

    registry.register(ToolSpec(
        name="sets_union",
        function=lambda args: _parse_set_operation(args, "union"),
        description="Computes union of two sets (elements in either set).",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "description": "First set as list"},
                "b": {"type": "array", "description": "Second set as list"},
            },
            "required": ["a", "b"]
        },
        returns="Union as sorted list",
        examples=[
            {"input": {"a": [1, 2, 3], "b": [2, 3, 4]}, "output": [1, 2, 3, 4]},
        ],
        domain="sets",
        tags=["union", "combine", "merge"],
    ))

    registry.register(ToolSpec(
        name="sets_intersection",
        function=lambda args: _parse_set_operation(args, "intersection"),
        description="Computes intersection of two sets (elements in both sets).",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "description": "First set as list"},
                "b": {"type": "array", "description": "Second set as list"},
            },
            "required": ["a", "b"]
        },
        returns="Intersection as sorted list",
        examples=[
            {"input": {"a": [1, 2, 3], "b": [2, 3, 4]}, "output": [2, 3]},
        ],
        domain="sets",
        tags=["intersection", "common", "overlap"],
    ))

    registry.register(ToolSpec(
        name="sets_subset",
        function=lambda args: _parse_set_operation(args, "subset"),
        description="Checks if first set is a subset of second set.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "description": "First set (potential subset)"},
                "b": {"type": "array", "description": "Second set (potential superset)"},
            },
            "required": ["a", "b"]
        },
        returns="Boolean: True if A ⊆ B",
        examples=[
            {"input": {"a": [1, 2], "b": [1, 2, 3, 4]}, "output": True},
            {"input": {"a": [1, 5], "b": [1, 2, 3, 4]}, "output": False},
        ],
        domain="sets",
        tags=["subset", "containment"],
    ))

    registry.register(ToolSpec(
        name="sets_complement",
        function=lambda args: _parse_set_operation(args, "complement"),
        description="Computes complement of first set relative to second set (universe).",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "description": "Set to complement"},
                "b": {"type": "array", "description": "Universe set"},
            },
            "required": ["a", "b"]
        },
        returns="Complement as sorted list",
        examples=[
            {"input": {"a": [1, 2, 3], "b": [1, 2, 3, 4, 5]}, "output": [4, 5]},
        ],
        domain="sets",
        tags=["complement", "difference"],
    ))
