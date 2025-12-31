"""
Operator induction for SCAN.

Given examples of a command pattern like "{primitive} around right",
induce a program that captures the operator semantics.

Key steps:
1. Parse commands to identify the pattern and extract inner actions
2. Convert command-level examples to IO pairs (action_seq -> output_seq)
3. Run enumerative search to find smallest consistent program
"""

import re
from typing import List, Tuple, Optional, Dict
from .dsl import Program, Seq
from .enumerate import find_smallest_consistent, find_consistent_programs


# Primitive to action token mapping
PRIMITIVE_TO_ACTION: Dict[str, str] = {
    "walk": "I_WALK",
    "run": "I_RUN",
    "jump": "I_JUMP",
    "look": "I_LOOK",
}


def primitive_to_action(primitive: str) -> List[str]:
    """Convert a primitive word to its action token sequence."""
    if primitive in PRIMITIVE_TO_ACTION:
        return [PRIMITIVE_TO_ACTION[primitive]]
    raise ValueError(f"Unknown primitive: {primitive}")


def parse_unary_command(command: str, operator_pattern: str) -> Optional[str]:
    """
    Parse a command matching a unary operator pattern.

    Args:
        command: e.g., "walk around right"
        operator_pattern: e.g., "around right" (the modifier)

    Returns:
        The primitive if command matches "{primitive} {operator_pattern}",
        otherwise None.
    """
    # Normalize
    command = command.strip().lower()
    operator_pattern = operator_pattern.strip().lower()

    # Check if command ends with the operator pattern
    if not command.endswith(operator_pattern):
        return None

    # Extract the prefix (should be a primitive)
    prefix = command[:-len(operator_pattern)].strip()

    if prefix in PRIMITIVE_TO_ACTION:
        return prefix

    return None


def extract_io_pairs(
    examples: List[Tuple[str, List[str]]],
    operator_pattern: str,
) -> List[Tuple[Seq, Seq]]:
    """
    Extract IO pairs from command-level examples.

    Args:
        examples: List of (command, output_tokens) pairs
        operator_pattern: The operator pattern (e.g., "around right")

    Returns:
        List of (input_action, output_tokens) pairs for the operator
    """
    io_pairs = []

    for command, output in examples:
        primitive = parse_unary_command(command, operator_pattern)
        if primitive is None:
            continue  # Skip non-matching commands

        # The input to the operator is the primitive's action
        input_action = primitive_to_action(primitive)
        io_pairs.append((input_action, list(output)))

    return io_pairs


def induce_unary_operator(
    examples: List[Tuple[str, List[str]]],
    operator_pattern: str,
    max_size: int = 12,
    verbose: bool = False,
) -> Optional[Program]:
    """
    Induce a unary operator from command-level examples.

    Args:
        examples: List of (command, output_tokens) pairs
        operator_pattern: The operator pattern to induce (e.g., "around right")
        max_size: Maximum program size for search
        verbose: Print debug information

    Returns:
        The induced program, or None if no consistent program found
    """
    # Extract IO pairs
    io_pairs = extract_io_pairs(examples, operator_pattern)

    if len(io_pairs) == 0:
        if verbose:
            print(f"No examples match pattern: {operator_pattern}")
        return None

    if verbose:
        print(f"Extracted {len(io_pairs)} IO pairs for '{operator_pattern}':")
        for inp, out in io_pairs:
            print(f"  {inp} -> {out[:8]}..." if len(out) > 8 else f"  {inp} -> {out}")

    # Find smallest consistent program
    program = find_smallest_consistent(io_pairs, max_size)

    if verbose:
        if program:
            print(f"Induced program: {program}")
            print(f"Program size: {program.size()}")
        else:
            print("No consistent program found")

    return program


def induce_and_verify(
    train_examples: List[Tuple[str, List[str]]],
    test_examples: List[Tuple[str, List[str]]],
    operator_pattern: str,
    max_size: int = 12,
    verbose: bool = True,
) -> Tuple[Optional[Program], float, float]:
    """
    Induce an operator and verify on train/test examples.

    Args:
        train_examples: Training examples (command, output)
        test_examples: Test examples (command, output)
        operator_pattern: Operator pattern to induce
        max_size: Maximum program size
        verbose: Print information

    Returns:
        (program, train_accuracy, test_accuracy)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Inducing operator: '{operator_pattern}'")
        print(f"Training examples: {len(train_examples)}")
        print(f"Test examples: {len(test_examples)}")
        print('='*60)

    # Induce from training examples
    program = induce_unary_operator(
        train_examples,
        operator_pattern,
        max_size=max_size,
        verbose=verbose,
    )

    if program is None:
        return None, 0.0, 0.0

    # Compute training accuracy (should be 100% by construction)
    train_io = extract_io_pairs(train_examples, operator_pattern)
    train_correct = 0
    for inp, expected in train_io:
        actual = program.execute(inp)
        if actual == expected:
            train_correct += 1
    train_acc = train_correct / len(train_io) if train_io else 0.0

    # Compute test accuracy
    test_io = extract_io_pairs(test_examples, operator_pattern)
    test_correct = 0
    for inp, expected in test_io:
        actual = program.execute(inp)
        if actual == expected:
            test_correct += 1
        elif verbose:
            print(f"\n  MISMATCH on {inp}:")
            print(f"    Expected: {expected}")
            print(f"    Got:      {actual}")
    test_acc = test_correct / len(test_io) if test_io else 0.0

    if verbose:
        print(f"\nResults:")
        print(f"  Train accuracy: {train_acc:.1%} ({train_correct}/{len(train_io)})")
        print(f"  Test accuracy:  {test_acc:.1%} ({test_correct}/{len(test_io)})")

    return program, train_acc, test_acc


class InducedOperatorLibrary:
    """
    A library of induced operators that can be used with an executor.

    This allows composing induced operators to solve more complex commands.
    """

    def __init__(self):
        self.operators: Dict[str, Program] = {}

    def add_operator(self, pattern: str, program: Program):
        """Add an induced operator to the library."""
        self.operators[pattern] = program

    def has_operator(self, pattern: str) -> bool:
        """Check if an operator has been induced."""
        return pattern in self.operators

    def apply_operator(self, pattern: str, action: Seq) -> Seq:
        """Apply an induced operator to an action sequence."""
        if pattern not in self.operators:
            raise ValueError(f"Unknown operator: {pattern}")
        return self.operators[pattern].execute(action)

    def __repr__(self):
        ops = ", ".join(self.operators.keys())
        return f"InducedOperatorLibrary([{ops}])"
