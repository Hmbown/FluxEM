"""
SCAN Rule Inducer: Discover rules from examples, then execute.

This module demonstrates the separation of rule discovery from rule execution.
Given a few examples of an operator's behavior, it induces a program that
captures the operator semantics, then uses that program to generalize to
unseen inputs.

This is FEW-SHOT RULE INDUCTION, not training-free execution.

Example:
    >>> from scan_inducer import induce_unary_operator
    >>> examples = [
    ...     ("walk around right", ["I_TURN_RIGHT", "I_WALK", ...]),
    ...     ("run around right", ["I_TURN_RIGHT", "I_RUN", ...]),
    ... ]
    >>> program = induce_unary_operator(examples, "around right")
    >>> program.execute(["I_JUMP"])  # Generalize to new primitive
    ['I_TURN_RIGHT', 'I_JUMP', 'I_TURN_RIGHT', 'I_JUMP', ...]

The key insight: once the rule is discovered (the hard part), execution
is deterministic and perfect (the easy part).
"""

from .dsl import (
    Program,
    Terminal,
    Concat,
    Repeat,
    Seq,
    LTURN,
    RTURN,
    ACTION,
    LTURN_TERM,
    RTURN_TERM,
    concat,
    repeat,
    pretty_print,
)

from .enumerate import (
    enumerate_programs,
    find_consistent_programs,
    find_smallest_consistent,
    compute_signature,
)

from .induce import (
    induce_unary_operator,
    induce_and_verify,
    extract_io_pairs,
    parse_unary_command,
    primitive_to_action,
    InducedOperatorLibrary,
)

__all__ = [
    # DSL
    "Program",
    "Terminal",
    "Concat",
    "Repeat",
    "Seq",
    "LTURN",
    "RTURN",
    "ACTION",
    "LTURN_TERM",
    "RTURN_TERM",
    "concat",
    "repeat",
    "pretty_print",
    # Enumeration
    "enumerate_programs",
    "find_consistent_programs",
    "find_smallest_consistent",
    "compute_signature",
    # Induction
    "induce_unary_operator",
    "induce_and_verify",
    "extract_io_pairs",
    "parse_unary_command",
    "primitive_to_action",
    "InducedOperatorLibrary",
]
