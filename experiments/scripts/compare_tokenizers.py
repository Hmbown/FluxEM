#!/usr/bin/env python3
"""Benchmark tokenization strategies for arithmetic.

Compares character-level, BPE-like, positional digit tokenization, and an
algebraic embedding approach.
"""

from __future__ import annotations

import sys
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

# Add parent path for imports
sys.path.insert(0, "/Volumes/VIXinSSD/FluxEM")

from fluxem.arithmetic.unified import create_unified_model


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = {
    "simple": {
        "expr": "5 + 3",
        "a": 5,
        "b": 3,
        "op": "+",
        "expected": 8,
        "difficulty": "Easy - single digits, no carry",
    },
    "carries": {
        "expr": "99 + 1",
        "a": 99,
        "b": 1,
        "op": "+",
        "expected": 100,
        "difficulty": "Medium - requires carry propagation (9+1=10, carry to next digit)",
    },
    "large_ood": {
        "expr": "999999 + 1",
        "a": 999999,
        "b": 1,
        "op": "+",
        "expected": 1000000,
        "difficulty": "Hard OOD - 6-digit carry chain, rarely seen in training",
    },
    "multi_digit": {
        "expr": "12345 + 67890",
        "a": 12345,
        "b": 67890,
        "op": "+",
        "expected": 80235,
        "difficulty": "Hard - 5-digit addition with multiple carries",
    },
}


# =============================================================================
# Abstract Tokenizer Base
# =============================================================================

@dataclass
class TokenizerResult:
    """Result from tokenizing an arithmetic expression."""
    tokens: List[Any]
    representation: str
    what_model_must_learn: str
    ood_capability: str
    computed_result: Optional[float] = None


class ArithmeticTokenizer(ABC):
    """Base class for arithmetic tokenization strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tokenization strategy."""
        pass

    @abstractmethod
    def tokenize(self, expr: str) -> TokenizerResult:
        """Tokenize an arithmetic expression."""
        pass

    @abstractmethod
    def can_compute(self) -> bool:
        """Whether this tokenizer can directly compute results."""
        pass

    def compute(self, a: float, b: float, op: str) -> Optional[float]:
        """Compute the result (if supported)."""
        return None


# =============================================================================
# 1. Character-Level Tokenizer
# =============================================================================

class CharacterTokenizer(ArithmeticTokenizer):
    """
    Character-level tokenization: each character is a separate token.

    Example: "123 + 456" -> ['1', '2', '3', ' ', '+', ' ', '4', '5', '6']

    This is how early transformer models processed text.
    """

    @property
    def name(self) -> str:
        return "Character-Level"

    def tokenize(self, expr: str) -> TokenizerResult:
        tokens = list(expr)

        return TokenizerResult(
            tokens=tokens,
            representation=f"[{', '.join(repr(t) for t in tokens)}]",
            what_model_must_learn=(
                "Must learn:\n"
                "  - Digit semantics (that '9' > '3')\n"
                "  - Place value (rightmost digit = ones, etc.)\n"
                "  - Carry rules (9+1 -> write 0, carry 1)\n"
                "  - Multi-digit propagation\n"
                "  - All from position-dependent patterns"
            ),
            ood_capability=(
                "POOR: Struggles with:\n"
                "  - Numbers longer than training max\n"
                "  - Carry chains longer than seen\n"
                "  - Digit combinations not in training"
            ),
        )

    def can_compute(self) -> bool:
        return False


# =============================================================================
# 2. BPE/WordPiece Style Tokenizer
# =============================================================================

class BPEStyleTokenizer(ArithmeticTokenizer):
    """
    BPE/WordPiece style: merges frequent character pairs.

    Example: "123" might become ['12', '3'] or ['123'] depending on vocabulary.

    This is how GPT, BERT, etc. tokenize text. Numbers are problematic because:
    - Common numbers like "100" might be one token
    - Rare numbers like "12847" split unpredictably
    - Same digit has different meanings in different token contexts
    """

    # Simulated BPE vocabulary (common number patterns)
    VOCAB = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "00", "10", "11", "12", "20", "50", "99", "100",
        "000", "999", "1000",
        "+", "-", "*", "/", " ",
    }

    @property
    def name(self) -> str:
        return "BPE/WordPiece"

    def tokenize(self, expr: str) -> TokenizerResult:
        # Simulate BPE tokenization
        tokens = self._bpe_tokenize(expr)

        return TokenizerResult(
            tokens=tokens,
            representation=f"[{', '.join(repr(t) for t in tokens)}]",
            what_model_must_learn=(
                "Must learn:\n"
                "  - Variable-length token semantics\n"
                "  - That '12' + '3' != '123' but = '15'\n"
                "  - Inconsistent tokenization of same number\n"
                "  - Token boundary effects on arithmetic\n"
                "  - 12847 vs 12 + 847 split differently"
            ),
            ood_capability=(
                "POOR-MEDIUM: Problems with:\n"
                "  - Numbers splitting at different boundaries\n"
                "  - Rare numbers have unusual tokenizations\n"
                "  - '99999' might be ['9', '99', '99'] or ['999', '99']"
            ),
        )

    def _bpe_tokenize(self, text: str) -> List[str]:
        """Simulate BPE tokenization (greedy longest match)."""
        tokens = []
        i = 0
        while i < len(text):
            # Try longest match first
            matched = False
            for length in [4, 3, 2, 1]:
                if i + length <= len(text):
                    candidate = text[i:i+length]
                    if candidate in self.VOCAB:
                        tokens.append(candidate)
                        i += length
                        matched = True
                        break
            if not matched:
                # Unknown character
                tokens.append(text[i])
                i += 1
        return tokens

    def can_compute(self) -> bool:
        return False


# =============================================================================
# 3. Digit-by-Digit with Position Tokenizer
# =============================================================================

class PositionalDigitTokenizer(ArithmeticTokenizer):
    """
    Digit-by-digit with explicit position encoding.

    Example: "123" -> [('1', pos=2), ('2', pos=1), ('3', pos=0)]

    Position encodes place value (10^pos).
    This is a step better - at least the model knows positions.
    """

    @property
    def name(self) -> str:
        return "Positional Digits"

    def tokenize(self, expr: str) -> TokenizerResult:
        tokens = []

        # Parse numbers with positions
        current_num = ""
        for char in expr:
            if char.isdigit():
                current_num += char
            else:
                if current_num:
                    # Add digit tokens with positions (leftmost digit has highest place value)
                    # "123" -> [('1', pos=2), ('2', pos=1), ('3', pos=0)]
                    for i, digit in enumerate(current_num):
                        pos = len(current_num) - 1 - i
                        tokens.append((digit, pos))
                    current_num = ""
                if char.strip():
                    tokens.append((char, None))  # Operators have no position

        # Handle trailing number
        if current_num:
            for i, digit in enumerate(current_num):
                pos = len(current_num) - 1 - i
                tokens.append((digit, pos))

        # Format for display
        repr_tokens = []
        for t in tokens:
            if t[1] is not None:
                repr_tokens.append(f"('{t[0]}', pos={t[1]})")
            else:
                repr_tokens.append(f"'{t[0]}'")

        return TokenizerResult(
            tokens=tokens,
            representation=f"[{', '.join(repr_tokens)}]",
            what_model_must_learn=(
                "Must learn:\n"
                "  - pos=N means multiply by 10^N\n"
                "  - Align positions for addition\n"
                "  - Carry logic between positions\n"
                "  - Output digit for each position\n"
                "  Better than char-level but still learns patterns"
            ),
            ood_capability=(
                "MEDIUM: Helps with:\n"
                "  - Position alignment\n"
                "  - Understanding place value\n"
                "  But still struggles with:\n"
                "  - Long carry chains\n"
                "  - Positions beyond training max"
            ),
        )

    def can_compute(self) -> bool:
        return False


# =============================================================================
# 4. FluxEM Algebraic Embedding
# =============================================================================

class FluxEMTokenizer(ArithmeticTokenizer):
    """
    FluxEM: Algebraic embeddings where operations are built-in.

    Example: "123" -> 128-dim embedding where:
        embed(a) + embed(b) = embed(a + b)  (exact algebraic identity)

    This is NOT learned - it's a mathematical property of the encoding.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.model = create_unified_model(dim=dim, linear_scale=1e7)

    @property
    def name(self) -> str:
        return "FluxEM Algebraic"

    def tokenize(self, expr: str) -> TokenizerResult:
        # Parse expression
        expr_clean = expr.replace(" ", "")

        # Find operator
        op_pos = -1
        op_char = None
        for i, c in enumerate(expr_clean):
            if c in "+-*/" and i > 0:
                op_pos = i
                op_char = c
                break

        if op_pos == -1:
            return TokenizerResult(
                tokens=[],
                representation="<parse error>",
                what_model_must_learn="N/A",
                ood_capability="N/A",
            )

        a_str = expr_clean[:op_pos]
        b_str = expr_clean[op_pos+1:]
        a = float(a_str)
        b = float(b_str)

        # Get embeddings
        emb_a = self.model.linear_encoder.encode_number(a)
        emb_b = self.model.linear_encoder.encode_number(b)

        # Compute result via embedding algebra
        if op_char == "+":
            result_emb = emb_a + emb_b
        elif op_char == "-":
            result_emb = emb_a - emb_b
        else:
            result_emb = emb_a  # For demo

        result = self.model.linear_encoder.decode(result_emb)

        return TokenizerResult(
            tokens=[f"embed({a_str})", op_char, f"embed({b_str})"],
            representation=(
                f"embed({a_str}) = {self.dim}-dim vector\n"
                f"    embed({b_str}) = {self.dim}-dim vector\n"
                f"    embed({a_str}) {op_char} embed({b_str}) = embed({a_str}{op_char}{b_str})"
            ),
            what_model_must_learn=(
                "Arithmetic constraints are defined by the encoder:\n"
                "  - embed(a) + embed(b) = embed(a + b) by construction\n"
                "  - No carry rules encoded in tokens\n"
                "  - Training focuses on routing and parsing"
            ),
            ood_capability=(
                "Behavior depends on parsing and scale limits:\n"
                "  - Supports magnitudes within configured scale\n"
                "  - Supports varying digit lengths within scale\n"
                "  - Uses algebraic encoder rather than digit patterns"
            ),
            computed_result=result,
        )

    def can_compute(self) -> bool:
        return True

    def compute(self, a: float, b: float, op: str) -> Optional[float]:
        expr = f"{a}{op}{b}="
        return self.model.compute(expr)


# =============================================================================
# Evaluator
# =============================================================================

class TokenizerEvaluator:
    """Evaluate and compare tokenization strategies."""

    def __init__(self):
        self.tokenizers = [
            CharacterTokenizer(),
            BPEStyleTokenizer(),
            PositionalDigitTokenizer(),
            FluxEMTokenizer(dim=128),
        ]

    def evaluate_case(self, case_name: str, case: Dict[str, Any]) -> Dict[str, TokenizerResult]:
        """Evaluate all tokenizers on a single test case."""
        results = {}
        for tokenizer in self.tokenizers:
            result = tokenizer.tokenize(case["expr"])
            if tokenizer.can_compute():
                result.computed_result = tokenizer.compute(
                    case["a"], case["b"], case["op"]
                )
            results[tokenizer.name] = result
        return results

    def run_benchmark(self):
        """Run the full benchmark."""
        print("table=tokenizer_benchmark")
        print("case\texpression\texpected\tdifficulty\ttokenizer\ttokens\trepresentation\twhat_model_must_learn\tood_capability\tcomputed_result\tcorrect\trel_error")

        for case_name, case in TEST_CASES.items():
            results = self.evaluate_case(case_name, case)
            self._emit_case_results(case_name, case, results)

    @staticmethod
    def _format_field(value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", "\\n").replace("\t", " ")

    def _emit_case_results(self, case_name: str, case: Dict[str, Any], results: Dict[str, TokenizerResult]):
        for name, result in results.items():
            computed_result = ""
            correct = ""
            rel_error = ""
            if result.computed_result is not None:
                expected = case["expected"]
                rel_error_val = abs(result.computed_result - expected) / max(abs(expected), 1)
                correct = "1" if rel_error_val < 0.01 else "0"
                rel_error = f"{rel_error_val:.6e}"
                computed_result = f"{result.computed_result:.6f}"

            row = [
                case_name,
                case["expr"],
                str(case["expected"]),
                case["difficulty"],
                name,
                self._format_field(" ".join(str(t) for t in result.tokens)),
                self._format_field(result.representation),
                self._format_field(result.what_model_must_learn),
                self._format_field(result.ood_capability),
                computed_result,
                correct,
                rel_error,
            ]
            print("\t".join(row))


# =============================================================================
# Simple OOD Accuracy Test
# =============================================================================

def test_ood_accuracy():
    """Test OOD accuracy of FluxEM vs simulated token-based model."""
    fluxem = FluxEMTokenizer(dim=128)

    # Test cases of increasing difficulty
    test_cases = [
        # In-distribution (assuming training on 0-999)
        ("ID: 2-digit", [(random.randint(10, 99), random.randint(10, 99)) for _ in range(10)]),
        ("ID: 3-digit", [(random.randint(100, 999), random.randint(100, 999)) for _ in range(10)]),
        # Out-of-distribution
        ("OOD: 4-digit", [(random.randint(1000, 9999), random.randint(1000, 9999)) for _ in range(10)]),
        ("OOD: 5-digit", [(random.randint(10000, 99999), random.randint(10000, 99999)) for _ in range(10)]),
        ("OOD: 6-digit", [(random.randint(100000, 999999), random.randint(100000, 999999)) for _ in range(10)]),
        # Adversarial: maximum carries
        ("Adversarial", [(99999, 1), (999999, 1), (9999999, 1), (99999999, 1), (999999999, 1)]),
    ]

    print("table=ood_accuracy")
    print("test_set\tfluxem_acc\ttoken_only_acc")

    for name, pairs in test_cases:
        fluxem_correct = 0

        for a, b in pairs:
            expected = a + b
            result = fluxem.compute(a, b, "+")
            if result is not None and abs(result - expected) < 0.5:
                fluxem_correct += 1

        fluxem_acc = fluxem_correct / len(pairs)

        # Simulated token-only accuracy
        # (degrades with number length, based on typical LLM behavior)
        if "ID" in name:
            token_acc = 0.95  # High on in-distribution
        elif "4-digit" in name:
            token_acc = 0.70  # Starts degrading
        elif "5-digit" in name:
            token_acc = 0.45  # Significant degradation
        elif "6-digit" in name:
            token_acc = 0.20  # Poor
        else:  # Adversarial
            token_acc = 0.05  # Very poor on max carry chains

        print(f"{name}\t{fluxem_acc:.6f}\t{token_acc:.6f}")

    print("table=ood_accuracy_notes")
    print("note\tvalue")
    print("token_only_accuracy\tsimulated")


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(42)

    evaluator = TokenizerEvaluator()
    evaluator.run_benchmark()

    test_ood_accuracy()

    return


if __name__ == "__main__":
    main()
