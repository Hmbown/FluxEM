"""
Bottom-up enumerative synthesis with observational equivalence pruning.

Key ideas:
1. Enumerate programs by increasing size (monotone cost search)
2. Prune by observational equivalence: if two programs produce the same
   outputs on all probe inputs, keep only the smaller one
3. Enforce hard constraints: programs must use ACTION, output length bounded
"""

from typing import Dict, List, Tuple, Optional, Set, Iterator
from .dsl import Program, Terminal, Concat, Repeat, Seq


# Probe inputs for computing signatures
# These are representative action sequences
DEFAULT_PROBES: List[Seq] = [
    ["I_WALK"],
    ["I_RUN"],
    ["I_LOOK"],
]

# Maximum output length to prevent explosion
MAX_OUTPUT_LENGTH = 64


def compute_signature(program: Program, probes: List[Seq]) -> Optional[Tuple[Tuple[str, ...], ...]]:
    """
    Compute the behavioral signature of a program.

    Returns a tuple of output sequences (as tuples) for each probe input.
    Returns None if any output exceeds MAX_OUTPUT_LENGTH.
    """
    outputs = []
    for probe in probes:
        try:
            result = program.execute(probe)
            if len(result) > MAX_OUTPUT_LENGTH:
                return None  # Too long, prune
            outputs.append(tuple(result))
        except Exception:
            return None  # Error during execution, prune
    return tuple(outputs)


def enumerate_programs(
    max_size: int,
    probes: List[Seq] = None,
    require_action: bool = True,
) -> Iterator[Program]:
    """
    Enumerate programs by increasing size with signature-based pruning.

    Uses bottom-up enumeration: first enumerate all programs of size 1,
    then size 2, etc. Programs with the same signature are deduplicated,
    keeping only the smallest.

    Args:
        max_size: Maximum program size to enumerate
        probes: Probe inputs for signature computation (default: DEFAULT_PROBES)
        require_action: If True, only yield programs that use ACTION

    Yields:
        Programs in order of increasing size, deduplicated by signature
    """
    if probes is None:
        probes = DEFAULT_PROBES

    # best_by_signature[sig] = (size, program)
    # We keep the smallest program for each signature
    best_by_signature: Dict[Tuple, Tuple[int, Program]] = {}

    # Programs by size for bottom-up enumeration
    # This includes ALL programs (even those not using ACTION) for building blocks
    programs_by_size: Dict[int, List[Program]] = {}

    def register_program(p: Program) -> Tuple[bool, bool]:
        """
        Register a program if it has a new or better signature.
        Returns (was_added_to_pool, should_yield).

        Programs are added to the pool even if they don't use ACTION,
        because they can be combined to form programs that do.
        """
        size = p.size()
        if size > max_size:
            return False, False

        sig = compute_signature(p, probes)
        if sig is None:
            return False, False  # Pruned (too long or error)

        # Check if we already have a program with this signature
        if sig in best_by_signature:
            existing_size, _ = best_by_signature[sig]
            if size >= existing_size:
                return False, False  # Not better

        # Register this program in the pool
        best_by_signature[sig] = (size, p)

        if size not in programs_by_size:
            programs_by_size[size] = []
        programs_by_size[size].append(p)

        # Only yield if it uses ACTION (or if require_action is False)
        should_yield = (not require_action) or p.uses_action()
        return True, should_yield

    # Size 1: Terminals (including LTURN, RTURN which don't use ACTION but are building blocks)
    programs_to_yield = []
    for name in ["ACTION", "LTURN", "RTURN"]:
        added, should_yield = register_program(Terminal(name))
        if should_yield:
            programs_to_yield.append(Terminal(name))

    # Yield size 1 programs that should be yielded
    for p in programs_to_yield:
        yield p

    # Bottom-up: build larger programs from smaller ones
    for target_size in range(2, max_size + 1):
        programs_to_yield = []

        # Concat: combine two smaller programs
        # Concat(left, right) has size = 1 + left.size() + right.size()
        # So left.size() + right.size() = target_size - 1
        for left_size in range(1, target_size - 1):
            right_size = target_size - 1 - left_size
            if right_size < 1:
                continue
            for left in programs_by_size.get(left_size, []):
                for right in programs_by_size.get(right_size, []):
                    p = Concat(left, right)
                    added, should_yield = register_program(p)
                    if should_yield:
                        programs_to_yield.append(p)

        # Repeat: wrap a smaller program
        # Repeat(n, body) has size = 2 + body.size()
        # So body.size() = target_size - 2
        body_size = target_size - 2
        if body_size >= 1:
            for body in programs_by_size.get(body_size, []):
                for n in [2, 3, 4]:
                    p = Repeat(n, body)
                    added, should_yield = register_program(p)
                    if should_yield:
                        programs_to_yield.append(p)

        # Yield new programs at this size
        for p in programs_to_yield:
            yield p


def find_consistent_programs(
    examples: List[Tuple[Seq, Seq]],
    max_size: int = 10,
    probes: List[Seq] = None,
) -> List[Tuple[int, Program]]:
    """
    Find all programs consistent with the given IO examples.

    Args:
        examples: List of (input, output) pairs
        max_size: Maximum program size to search
        probes: Probe inputs for signature pruning

    Returns:
        List of (size, program) tuples, sorted by size
    """
    consistent = []

    for program in enumerate_programs(max_size, probes):
        # Check if program is consistent with all examples
        is_consistent = True
        for input_seq, expected_output in examples:
            try:
                actual_output = program.execute(input_seq)
                if actual_output != expected_output:
                    is_consistent = False
                    break
            except Exception:
                is_consistent = False
                break

        if is_consistent:
            consistent.append((program.size(), program))

    # Sort by size (MDL: prefer smaller programs)
    consistent.sort(key=lambda x: x[0])
    return consistent


def find_smallest_consistent(
    examples: List[Tuple[Seq, Seq]],
    max_size: int = 10,
    probes: List[Seq] = None,
) -> Optional[Program]:
    """
    Find the smallest program consistent with all examples (MDL).

    Args:
        examples: List of (input, output) pairs
        max_size: Maximum program size to search
        probes: Probe inputs for signature pruning

    Returns:
        The smallest consistent program, or None if no program found
    """
    consistent = find_consistent_programs(examples, max_size, probes)
    if consistent:
        return consistent[0][1]  # Return the smallest
    return None
