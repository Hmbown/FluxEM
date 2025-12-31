"""
FluxEM Compositional: Oracle baseline for SCAN.

This module provides an oracle baseline that separates rule discovery from
rule execution. It encodes SCAN's compositional semantics directly, demonstrating
that once rules are known, composition is trivial.

This is NOT a learning result. It's a diagnostic.

Example:
    >>> from fluxem.compositional import AlgebraicSCANSolver
    >>> solver = AlgebraicSCANSolver()
    >>> solver.solve("jump twice")
    'I_JUMP I_JUMP'
    >>> solver.solve("walk around left")
    'I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK'

The Diagnostic:
    SCAN benchmarks conflate rule discovery with rule execution. Neural networks
    fail at rule discovery from limited examples, not at applying known rules.
    This oracle achieves 100% because it encodes the rules directly.

See docs/SCAN_BASELINE.md for full framing.
"""

from .algebra import AlgebraicSCANSolver, evaluate_accuracy
from .scan_loader import load_scan_split, load_scan_file, get_split_stats
from .eval import evaluate_algebraic, print_results, SPLITS

__all__ = [
    "AlgebraicSCANSolver",
    "evaluate_accuracy",
    "evaluate_algebraic",
    "print_results",
    "SPLITS",
    "load_scan_split",
    "load_scan_file",
    "get_split_stats",
]
