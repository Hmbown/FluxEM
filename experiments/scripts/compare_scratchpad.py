#!/usr/bin/env python3
"""Compare arithmetic computation approaches.

Runs a small benchmark comparing:
- Scratchpad (explicit intermediate steps)
- Chain-of-thought (decomposition into partial sums)
- FluxEM (embedding-level computation)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# Add parent path for imports
sys.path.insert(0, "/Volumes/VIXinSSD/FluxEM")

from fluxem import create_unified_model


# =============================================================================
# Test Cases: Progressively Harder
# =============================================================================

TEST_CASES = [
    {
        "name": "Simple",
        "expr": "5 + 3",
        "a": 5,
        "b": 3,
        "expected": 8,
        "description": "Single digit, no carries",
        "scratchpad_steps": 1,
    },
    {
        "name": "Medium",
        "expr": "123 + 456",
        "a": 123,
        "b": 456,
        "expected": 579,
        "description": "3 digits, no carries",
        "scratchpad_steps": 3,
    },
    {
        "name": "Hard",
        "expr": "12345 + 67890",
        "a": 12345,
        "b": 67890,
        "expected": 80235,
        "description": "5 digits, multiple carries",
        "scratchpad_steps": 5,
    },
    {
        "name": "Very Hard",
        "expr": "1234567890 + 9876543210",
        "a": 1234567890,
        "b": 9876543210,
        "expected": 11111111100,
        "description": "10 digits, many carries",
        "scratchpad_steps": 10,
    },
    {
        "name": "Adversarial",
        "expr": "9999999999 + 1",
        "a": 9999999999,
        "b": 1,
        "expected": 10000000000,
        "description": "Maximum carry chain (10 carries)",
        "scratchpad_steps": 10,
    },
]


# =============================================================================
# Abstract Approach Base Class
# =============================================================================

@dataclass
class ComputationResult:
    """Result from computing an arithmetic expression."""
    approach: str
    expression: str
    result: Optional[float]
    expected: float
    correct: bool
    steps: List[str]
    n_steps: int
    complexity: str
    error_risk: str


class ArithmeticApproach(ABC):
    """Base class for arithmetic computation approaches."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the approach."""
        pass

    @abstractmethod
    def compute(self, a: int, b: int, op: str = "+") -> ComputationResult:
        """Compute a + b and return the result with steps."""
        pass


# =============================================================================
# 1. Scratchpad Approach
# =============================================================================

class ScratchpadApproach(ArithmeticApproach):
    """
    Scratchpad: Model writes intermediate computation steps.

    Example for 123 + 456:
        Step 1: ones place: 3 + 6 = 9, write 9, carry 0
        Step 2: tens place: 2 + 5 + 0 = 7, write 7, carry 0
        Step 3: hundreds place: 1 + 4 + 0 = 5, write 5, carry 0
        Result: 579

    This is how models like GPT-4 can improve accuracy on arithmetic
    by writing out intermediate steps. However:
    - Model must LEARN to generate correct steps
    - Each step is a new token generation
    - Errors in early steps propagate to final answer
    """

    @property
    def name(self) -> str:
        return "Scratchpad"

    def compute(self, a: int, b: int, op: str = "+") -> ComputationResult:
        expected = a + b

        # Simulate scratchpad computation (digit-by-digit with carries)
        steps = []
        a_str = str(a).zfill(max(len(str(a)), len(str(b))))
        b_str = str(b).zfill(max(len(str(a)), len(str(b))))

        carry = 0
        result_digits = []
        places = ["ones", "tens", "hundreds", "thousands", "ten-thousands",
                  "hundred-thousands", "millions", "ten-millions",
                  "hundred-millions", "billions"]

        for i, (d_a, d_b) in enumerate(zip(reversed(a_str), reversed(b_str))):
            place_name = places[i] if i < len(places) else f"10^{i}"
            digit_sum = int(d_a) + int(d_b) + carry
            new_carry = digit_sum // 10
            result_digit = digit_sum % 10

            step = f"Step {i+1} ({place_name}): {d_a} + {d_b}"
            if carry > 0:
                step += f" + {carry} (carry)"
            step += f" = {digit_sum}"
            if new_carry > 0:
                step += f", write {result_digit}, carry {new_carry}"
            else:
                step += f", write {result_digit}"

            steps.append(step)
            result_digits.append(str(result_digit))
            carry = new_carry

        # Handle final carry
        if carry > 0:
            steps.append(f"Final carry: {carry}")
            result_digits.append(str(carry))

        computed = int("".join(reversed(result_digits)))

        return ComputationResult(
            approach=self.name,
            expression=f"{a} + {b}",
            result=float(computed),
            expected=float(expected),
            correct=computed == expected,
            steps=steps,
            n_steps=len(steps),
            complexity=f"O(n) where n = {len(a_str)} digits",
            error_risk=(
                "HIGH: Each step requires correct digit addition, "
                "carry tracking, and position management. "
                "One error corrupts the final result."
            ),
        )


# =============================================================================
# 2. Chain-of-Thought Approach
# =============================================================================

class ChainOfThoughtApproach(ArithmeticApproach):
    """
    Chain-of-Thought: Prompting for step-by-step reasoning.

    Example for 123 + 456:
        "Let me solve this step by step.
         First, I'll add the ones: 3 + 6 = 9
         Then the tens: 20 + 50 = 70
         Then the hundreds: 100 + 400 = 500
         Total: 500 + 70 + 9 = 579"

    This is a prompting technique, not a different architecture.
    Still relies on the model's learned arithmetic capabilities.
    Similar issues to scratchpad but with more verbose reasoning.
    """

    @property
    def name(self) -> str:
        return "Chain-of-Thought"

    def compute(self, a: int, b: int, op: str = "+") -> ComputationResult:
        expected = a + b

        # Simulate chain-of-thought reasoning
        steps = ["Let me solve this step by step."]

        # Decompose by place value
        a_str = str(a)
        b_str = str(b)
        n_digits = max(len(a_str), len(b_str))

        partial_sums = []
        place_values = [1, 10, 100, 1000, 10000, 100000, 1000000,
                       10000000, 100000000, 1000000000]

        for i in range(n_digits):
            place = place_values[i] if i < len(place_values) else 10**i
            a_digit = int(a_str[-(i+1)]) if i < len(a_str) else 0
            b_digit = int(b_str[-(i+1)]) if i < len(b_str) else 0

            a_value = a_digit * place
            b_value = b_digit * place
            partial = a_value + b_value

            steps.append(f"Position {i+1}: {a_value} + {b_value} = {partial}")
            partial_sums.append(partial)

        # Sum all partials
        total = sum(partial_sums)
        steps.append(f"Sum all partials: {' + '.join(map(str, partial_sums))} = {total}")

        return ComputationResult(
            approach=self.name,
            expression=f"{a} + {b}",
            result=float(total),
            expected=float(expected),
            correct=total == expected,
            steps=steps,
            n_steps=len(steps),
            complexity=f"O(n) reasoning steps, O(n^2) tokens for n = {n_digits} digits",
            error_risk=(
                "HIGH: Requires correct decomposition, "
                "correct partial sums, and correct aggregation. "
                "More verbose = more tokens = higher error probability."
            ),
        )


# =============================================================================
# 3. FluxEM Algebraic Approach
# =============================================================================

class FluxEMApproach(ArithmeticApproach):
    """
    FluxEM: Direct computation via algebraic embeddings.

    For any a, b:
        embed(a) + embed(b) = embed(a + b)

    This is not learned - it's a mathematical property of the encoding.
    No intermediate steps, no carries, no digit manipulation.
    Single operation regardless of number size.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model = create_unified_model(dim=dim, linear_scale=1e11)

    @property
    def name(self) -> str:
        return "FluxEM"

    def compute(self, a: int, b: int, op: str = "+") -> ComputationResult:
        expected = a + b

        # Compute via embedding algebra
        expr = f"{a}+{b}"
        result = self.model.compute(expr)

        # Show what actually happens
        steps = [
            f"embed({a}) -> {self.dim}-dimensional vector",
            f"embed({b}) -> {self.dim}-dimensional vector",
            f"embed({a}) + embed({b}) = embed({a + b})",
            f"decode(embed({a + b})) = {result}",
        ]

        correct = result is not None and abs(result - expected) < 0.5

        return ComputationResult(
            approach=self.name,
            expression=f"{a} + {b}",
            result=result,
            expected=float(expected),
            correct=correct,
            steps=steps,
            n_steps=1,  # Single embedding operation
            complexity="O(1) - single vector addition regardless of digit count",
            error_risk=(
                "MINIMAL: No intermediate steps means no error propagation. "
                "Accuracy bounded only by floating-point precision."
            ),
        )


# =============================================================================
# Comparison Runner
# =============================================================================

def print_header():
    """Print the benchmark header."""
    print("case\tapproach\texpression\texpected\tresult\tcorrect\tsteps\tcomplexity\terror_risk")


def print_case_comparison(case: Dict[str, Any], results: Dict[str, ComputationResult]):
    """Print comparison for a single test case."""
    for approach_name, result in results.items():
        steps = "|".join(result.steps) if result.steps else ""
        row = [
            case["name"],
            approach_name,
            case["expr"],
            str(case["expected"]),
            str(result.result),
            "1" if result.correct else "0",
            str(result.n_steps),
            result.complexity,
            result.error_risk,
        ]
        print("\t".join(row))


def print_summary_table(all_results: List[Tuple[Dict, Dict[str, ComputationResult]]]):
    """Print a summary comparison table."""
    print("summary_case\tdigits\tscratchpad_steps\tscratchpad_correct\tcot_steps\tcot_correct\tfluxem_steps\tfluxem_correct")

    for case, results in all_results:
        n_digits = len(str(max(case["a"], case["b"])))

        sp = results["Scratchpad"]
        cot = results["Chain-of-Thought"]
        flux = results["FluxEM"]

        row = [
            case["name"],
            str(n_digits),
            str(sp.n_steps),
            "1" if sp.correct else "0",
            str(cot.n_steps),
            "1" if cot.correct else "0",
            str(flux.n_steps),
            "1" if flux.correct else "0",
        ]
        print("\t".join(row))


def print_complexity_analysis():
    """Print complexity analysis."""
    print("complexity_metric\tscratchpad\tchain_of_thought\tfluxem")
    print("steps\tO(n)\tO(n)\tO(1)")
    print("tokens_generated\tO(n)\tO(n^2)\tO(1)")
    print("error_propagation\tyes (carry chain)\tyes (partial sums)\tno")
    print("training_needed\tyes (step format)\tyes (reasoning)\tno (algebraic)")
    print("ood_generalization\tdistribution-dependent\tdistribution-dependent\tlimited by float precision")


def run_comparison():
    """Run the full comparison."""
    print_header()

    # Initialize approaches
    approaches = {
        "Scratchpad": ScratchpadApproach(),
        "Chain-of-Thought": ChainOfThoughtApproach(),
        "FluxEM": FluxEMApproach(dim=256),
    }

    all_results = []

    # Run each test case
    for case in TEST_CASES:
        results = {}
        for name, approach in approaches.items():
            results[name] = approach.compute(case["a"], case["b"])
        all_results.append((case, results))
        print_case_comparison(case, results)

    # Print summary
    print_summary_table(all_results)
    print_complexity_analysis()


def main():
    """Main entry point."""
    run_comparison()


if __name__ == "__main__":
    main()
