"""Music domain - pitch class sets, atonal theory, chord identification.

This module provides deterministic music theory computations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Chord Patterns
# =============================================================================

CHORD_PATTERNS = {
    (0, 3, 7): "minor triad",
    (0, 4, 7): "major triad",
    (0, 3, 6): "diminished triad",
    (0, 4, 8): "augmented triad",
    (0, 4, 7, 10): "dominant 7th",
    (0, 4, 7, 11): "major 7th",
    (0, 3, 7, 10): "minor 7th",
    (0, 3, 6, 9): "diminished 7th",
    (0, 3, 6, 10): "half-diminished 7th",
    (0, 4, 7, 9): "major 6th",
    (0, 3, 7, 9): "minor 6th",
    (0, 2, 7): "sus2",
    (0, 5, 7): "sus4",
    (0, 4, 7, 10, 14): "9th",
    (0, 2, 4, 5, 7, 9, 11): "major scale",
    (0, 2, 3, 5, 7, 8, 10): "natural minor scale",
}


# =============================================================================
# Core Functions
# =============================================================================

def normal_form(pcs: List[int]) -> List[int]:
    """Compute normal form (most compact left-packed rotation) of pitch class set.

    Args:
        pcs: Pitch class set as list of integers 0-11

    Returns:
        Normal form as sorted pitch class list
    """
    if not pcs:
        return []

    # Reduce to unique pitch classes mod 12
    pcs = sorted(set(p % 12 for p in pcs))
    n = len(pcs)

    if n == 1:
        return [0]

    # Try all rotations
    best = None
    for i in range(n):
        rotation = [(pcs[(i + j) % n] - pcs[i]) % 12 for j in range(n)]
        if best is None or rotation < best:
            best = rotation

    return best


def prime_form(pcs: List[int]) -> List[int]:
    """Compute prime form (most compact representation) of pitch class set.

    Args:
        pcs: Pitch class set as list of integers 0-11

    Returns:
        Prime form as list starting with 0
    """
    if not pcs:
        return []

    nf = normal_form(pcs)
    # Also try inversion
    inverted = [(12 - p) % 12 for p in pcs]
    nf_inv = normal_form(inverted)

    # Return the more compact form
    return nf if nf <= nf_inv else nf_inv


def transposition(pcs: List[int], n: int) -> List[int]:
    """Transpose a pitch class set by n semitones.

    Args:
        pcs: Pitch class set as list of integers 0-11
        n: Number of semitones to transpose

    Returns:
        Transposed pitch class set
    """
    return sorted((p + n) % 12 for p in pcs)


def identify_chord(pcs: List[int]) -> str:
    """Identify chord quality from pitch class set.

    Args:
        pcs: Pitch class set as list of integers 0-11

    Returns:
        Chord quality name or 'unknown'
    """
    pf = tuple(prime_form(pcs))
    return CHORD_PATTERNS.get(pf, "unknown chord type")


def interval_class(p1: int, p2: int) -> int:
    """Compute interval class between two pitch classes.

    Args:
        p1: First pitch class (0-11)
        p2: Second pitch class (0-11)

    Returns:
        Interval class (0-6)
    """
    diff = abs(p2 - p1) % 12
    return min(diff, 12 - diff)


def interval_vector(pcs: List[int]) -> List[int]:
    """Compute interval vector (interval content) of a pitch class set.

    Args:
        pcs: Pitch class set

    Returns:
        6-element interval vector [ic1, ic2, ic3, ic4, ic5, ic6]
    """
    pcs = sorted(set(p % 12 for p in pcs))
    vector = [0] * 6

    for i, p1 in enumerate(pcs):
        for p2 in pcs[i + 1:]:
            ic = interval_class(p1, p2)
            if 1 <= ic <= 6:
                vector[ic - 1] += 1

    return vector


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_pcs(args) -> List[int]:
    if isinstance(args, dict):
        pcs = args.get("pcs", args.get("pitches", args.get("notes")))
        if pcs is not None:
            return [int(p) for p in pcs]
        return [int(p) for p in list(args.values())[0]]
    if isinstance(args, (list, tuple)):
        return [int(p) for p in args]
    raise ValueError(f"Cannot parse pitch class set: {args}")


def _parse_transpose_args(args) -> Tuple[List[int], int]:
    if isinstance(args, dict):
        pcs = args.get("pcs", args.get("pitches"))
        n = args.get("n", args.get("semitones", args.get("interval")))
        return [int(p) for p in pcs], int(n)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [int(p) for p in args[0]], int(args[1])
    raise ValueError(f"Cannot parse transpose args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register music theory tools in the registry."""

    registry.register(ToolSpec(
        name="music_prime_form",
        function=lambda args: prime_form(_parse_pcs(args)),
        description="Computes prime form (most compact representation) of a pitch class set.",
        parameters={
            "type": "object",
            "properties": {
                "pcs": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Pitch class set as integers 0-11"
                }
            },
            "required": ["pcs"]
        },
        returns="Prime form as list starting with 0",
        examples=[
            {"input": {"pcs": [0, 4, 7]}, "output": [0, 3, 7]},
            {"input": {"pcs": [2, 6, 9]}, "output": [0, 3, 7]},
        ],
        domain="music",
        tags=["atonal", "prime form", "pitch class"],
    ))

    registry.register(ToolSpec(
        name="music_normal_form",
        function=lambda args: normal_form(_parse_pcs(args)),
        description="Computes normal form (most compact left-packed rotation) of a pitch class set.",
        parameters={
            "type": "object",
            "properties": {
                "pcs": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Pitch class set as integers 0-11"
                }
            },
            "required": ["pcs"]
        },
        returns="Normal form as sorted list",
        examples=[
            {"input": {"pcs": [7, 0, 4]}, "output": [0, 4, 7]},
        ],
        domain="music",
        tags=["atonal", "normal form", "pitch class"],
    ))

    registry.register(ToolSpec(
        name="music_chord_type",
        function=lambda args: identify_chord(_parse_pcs(args)),
        description="Identifies chord quality from a pitch class set.",
        parameters={
            "type": "object",
            "properties": {
                "pcs": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Pitch class set as integers 0-11"
                }
            },
            "required": ["pcs"]
        },
        returns="Chord quality name (e.g., 'major triad')",
        examples=[
            {"input": {"pcs": [0, 4, 7]}, "output": "major triad"},
            {"input": {"pcs": [0, 3, 7]}, "output": "minor triad"},
        ],
        domain="music",
        tags=["chord", "identification", "harmony"],
    ))

    registry.register(ToolSpec(
        name="music_transpose",
        function=lambda args: transposition(*_parse_transpose_args(args)),
        description="Transposes a pitch class set by n semitones.",
        parameters={
            "type": "object",
            "properties": {
                "pcs": {"type": "array", "items": {"type": "integer"}, "description": "Pitch class set"},
                "n": {"type": "integer", "description": "Semitones to transpose"},
            },
            "required": ["pcs", "n"]
        },
        returns="Transposed pitch class set",
        examples=[
            {"input": {"pcs": [0, 4, 7], "n": 7}, "output": [2, 7, 11]},
        ],
        domain="music",
        tags=["transpose", "pitch class", "transposition"],
    ))
