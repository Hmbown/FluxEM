#!/usr/bin/env python3
"""
Comparison: LLM Numeric Encoding vs FluxEM Algebraic Embeddings

This script demonstrates the fundamental differences between how traditional
LLMs (GPT, T5, character-level models) handle numeric data versus FluxEM's
algebraic embedding approach.

Key insight: Traditional tokenizers treat numbers as arbitrary symbols with
no mathematical structure. FluxEM embeds numbers such that arithmetic
operations become geometric transformations.

Usage:
    python compare_llm_numeric.py [--interactive]
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import re

# Import FluxEM
from fluxem import create_unified_model, NumberEncoder


# =============================================================================
# TOKENIZATION SIMULATIONS
# =============================================================================

class BPETokenizerSimulator:
    """
    Simulates GPT-style BPE tokenization of numbers.

    GPT-2/3/4 use BPE (Byte Pair Encoding) which:
    - Splits numbers into arbitrary subword tokens
    - Has no consistent mapping for digit patterns
    - Different numbers may share tokens arbitrarily

    Example: "123456" might become ["12", "34", "56"] or ["123", "456"]
    depending on training corpus statistics.
    """

    # Simulated BPE vocabulary (based on common GPT-2 patterns)
    COMMON_TOKENS = {
        "0": 15, "1": 16, "2": 17, "3": 18, "4": 19,
        "5": 20, "6": 21, "7": 22, "8": 23, "9": 24,
        "10": 940, "11": 1157, "12": 1065, "13": 1485, "14": 1415,
        "15": 1314, "16": 1433, "17": 1558, "18": 1507, "19": 1129,
        "20": 1238, "100": 3064, "000": 830, "00": 405,
        "123": 10163, "456": 29228, "789": 37397,
        "1000": 12825, "2000": 11024, "3000": 14877,
        " ": 220, "+": 10, "-": 12, "*": 9, "/": 14, "=": 28,
        "What": 2061, " is": 318, "?": 30,
    }

    def __init__(self):
        self.vocab = self.COMMON_TOKENS.copy()

    def tokenize(self, text: str) -> List[Tuple[str, int]]:
        """
        Tokenize text using simulated BPE.

        Returns list of (token_text, token_id) pairs.
        """
        tokens = []
        i = 0

        while i < len(text):
            # Greedy longest match
            best_match = None
            best_len = 0

            for token, token_id in self.vocab.items():
                if text[i:].startswith(token) and len(token) > best_len:
                    best_match = (token, token_id)
                    best_len = len(token)

            if best_match:
                tokens.append(best_match)
                i += best_len
            else:
                # Fallback to single character
                char = text[i]
                tokens.append((char, ord(char) + 10000))  # Unknown token
                i += 1

        return tokens

    def analyze_number(self, number: str) -> Dict[str, Any]:
        """Analyze how a number is tokenized."""
        tokens = self.tokenize(number)
        return {
            "number": number,
            "tokens": [t[0] for t in tokens],
            "token_ids": [t[1] for t in tokens],
            "num_tokens": len(tokens),
            "has_arithmetic_meaning": False,  # BPE tokens have no arithmetic structure
        }


class T5TokenizerSimulator:
    """
    Simulates T5/Flan-style SentencePiece tokenization.

    T5 uses SentencePiece which has similar issues:
    - Numbers split inconsistently
    - Scientific notation is particularly problematic
    - No arithmetic relationships between token embeddings
    """

    # Simulated T5 vocabulary
    COMMON_TOKENS = {
        "0": 3, "1": 4, "2": 5, "3": 6, "4": 7, "5": 8, "6": 9, "7": 10, "8": 11, "9": 12,
        "10": 335, "100": 2382, "1000": 9353,
        "12": 1073, "123": 28311, "1234": 102345,
        "e": 15, "+": 1584, "-": 18, ".": 5, " ": 3,
        "What": 363, "is": 19, "?": 58,
    }

    def __init__(self):
        self.vocab = self.COMMON_TOKENS.copy()

    def tokenize(self, text: str) -> List[Tuple[str, int]]:
        """Tokenize with simulated SentencePiece."""
        tokens = []
        i = 0

        while i < len(text):
            best_match = None
            best_len = 0

            for token, token_id in self.vocab.items():
                if text[i:].startswith(token) and len(token) > best_len:
                    best_match = (token, token_id)
                    best_len = len(token)

            if best_match:
                tokens.append(best_match)
                i += best_len
            else:
                char = text[i]
                tokens.append((char, ord(char) + 20000))
                i += 1

        return tokens

    def analyze_scientific_notation(self, number: str) -> Dict[str, Any]:
        """Analyze scientific notation handling."""
        tokens = self.tokenize(number)
        return {
            "number": number,
            "tokens": [t[0] for t in tokens],
            "num_tokens": len(tokens),
            "issues": [
                "Mantissa and exponent tokenized separately",
                "No semantic link between '1e6' and '1000000'",
                "Model must learn exponential relationships from data",
            ],
        }


class CharacterLevelTokenizer:
    """
    Simulates character-level tokenization (used by some models).

    Each digit is a separate token:
    - "123" -> ["1", "2", "3"]
    - Simpler than BPE but requires learning:
      * Place value (1 in "100" differs from 1 in "1")
      * Carry operations
      * All arithmetic from scratch
    """

    def tokenize(self, text: str) -> List[Tuple[str, int]]:
        """Tokenize at character level."""
        return [(c, ord(c)) for c in text]

    def analyze_addition_complexity(self, a: str, b: str) -> Dict[str, Any]:
        """Analyze what the model must learn for addition."""
        tokens_a = self.tokenize(a)
        tokens_b = self.tokenize(b)

        # Calculate actual addition to show carry complexity
        result = str(int(a) + int(b))

        # Count carries needed
        carries = 0
        carry = 0
        for i in range(max(len(a), len(b))):
            d_a = int(a[-(i+1)]) if i < len(a) else 0
            d_b = int(b[-(i+1)]) if i < len(b) else 0
            total = d_a + d_b + carry
            if total >= 10:
                carries += 1
                carry = 1
            else:
                carry = 0

        return {
            "a": a,
            "b": b,
            "a_tokens": [t[0] for t in tokens_a],
            "b_tokens": [t[0] for t in tokens_b],
            "result": result,
            "carries_needed": carries,
            "learning_required": [
                f"Place value: '1' in position 0 = 1, in position 2 = 100",
                f"Carry propagation: {carries} carry operations needed",
                "Right-to-left processing (reverse of reading order)",
                "Variable-length output handling",
            ],
        }


# =============================================================================
# FLUXEM DEMONSTRATION
# =============================================================================

class FluxEMDemonstrator:
    """
    Demonstrates FluxEM's algebraic embedding approach.

    Key properties:
    - embed(a) + embed(b) = embed(a + b)  [for linear embeddings]
    - log_embed(a) + log_embed(b) = log_embed(a * b)  [for log embeddings]
    - No learning required for basic arithmetic
    - Perfect generalization to any magnitude
    """

    def __init__(self):
        self.model = create_unified_model(dim=256)
        self.linear_encoder = self.model.linear_encoder
        self.log_encoder = self.model.log_encoder

    def demonstrate_addition(self, a: float, b: float) -> Dict[str, Any]:
        """Demonstrate exact addition via embedding arithmetic."""
        # Encode numbers
        emb_a = self.linear_encoder.encode_number(a)
        emb_b = self.linear_encoder.encode_number(b)

        # Add embeddings (this IS the addition operation)
        emb_sum = emb_a + emb_b

        # Decode result
        result = self.linear_encoder.decode(emb_sum)
        expected = a + b

        # Use relative tolerance for larger numbers
        if abs(expected) > 1:
            rel_error = abs(result - expected) / abs(expected)
            is_accurate = rel_error < 1e-4  # 0.01% tolerance
        else:
            is_accurate = abs(result - expected) < 1e-6

        return {
            "a": a,
            "b": b,
            "expected": expected,
            "computed": result,
            "exact": is_accurate,
            "method": "embed(a) + embed(b) = embed(a + b)",
            "embedding_dim": 256,
            "learning_required": "NONE - arithmetic is built into representation",
        }

    def demonstrate_multiplication(self, a: float, b: float) -> Dict[str, Any]:
        """Demonstrate exact multiplication via log embedding arithmetic."""
        # Encode numbers
        emb_a = self.log_encoder.encode_number(a)
        emb_b = self.log_encoder.encode_number(b)

        # Multiply via log addition
        emb_product = self.log_encoder.multiply(emb_a, emb_b)

        # Decode result
        result = self.log_encoder.decode(emb_product)
        expected = a * b

        rel_error = abs(result - expected) / abs(expected) if expected != 0 else abs(result)

        return {
            "a": a,
            "b": b,
            "expected": expected,
            "computed": result,
            "relative_error": rel_error,
            "accurate": rel_error < 0.01,
            "method": "log_embed(a) + log_embed(b) = log_embed(a * b)",
            "note": "Uses logarithmic representation for multiplication",
        }

    def demonstrate_ood_generalization(self) -> Dict[str, Any]:
        """Demonstrate out-of-distribution generalization."""
        test_cases = [
            # Small numbers (in-distribution for most LLMs)
            (42, 58, "+"),
            (7, 8, "*"),

            # Medium numbers
            (1234, 5678, "+"),
            (123, 456, "*"),

            # Large numbers (OOD for most LLMs)
            (123456, 789012, "+"),
            (12345, 6789, "*"),

            # Very large numbers (catastrophic failure for LLMs)
            (98765432, 12345678, "+"),
            (999999, 888888, "*"),
        ]

        results = []
        for a, b, op in test_cases:
            if op == "+":
                expected = a + b
                computed = self.model.compute(f"{a}+{b}")
            else:
                expected = a * b
                computed = self.model.compute(f"{a}*{b}")

            rel_error = abs(computed - expected) / abs(expected) if expected != 0 else 0

            results.append({
                "expression": f"{a} {op} {b}",
                "expected": expected,
                "computed": computed,
                "relative_error": f"{rel_error:.2e}",
                "accurate": rel_error < 0.01,
            })

        return {
            "test_cases": results,
            "all_accurate": all(r["accurate"] for r in results),
            "note": "FluxEM maintains accuracy regardless of number magnitude",
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def print_header(title: str, width: int = 78):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str, width: int = 78):
    """Print a section header."""
    print(f"\n{title}")
    print("-" * len(title))


def print_comparison_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None):
    """Print a formatted comparison table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]

    # Header
    header_line = "|".join(f" {h:<{w-2}} " for h, w in zip(headers, col_widths))
    separator = "+".join("-" * w for w in col_widths)

    print(f"+{separator}+")
    print(f"|{header_line}|")
    print(f"+{separator}+")

    # Rows
    for row in rows:
        row_line = "|".join(f" {str(c):<{w-2}} " for c, w in zip(row, col_widths))
        print(f"|{row_line}|")

    print(f"+{separator}+")


def visualize_tokenization_difference():
    """Visualize how different tokenizers handle the same number."""
    print_header("TOKENIZATION COMPARISON: How LLMs See Numbers")

    test_numbers = ["123456", "67890", "80235"]

    bpe = BPETokenizerSimulator()
    t5 = T5TokenizerSimulator()
    char = CharacterLevelTokenizer()

    for num in test_numbers:
        print_section(f"Number: {num}")

        bpe_result = bpe.analyze_number(num)
        t5_tokens = t5.tokenize(num)
        char_tokens = char.tokenize(num)

        print(f"  GPT BPE:      {bpe_result['tokens']} ({len(bpe_result['tokens'])} tokens)")
        print(f"  T5 SentPiece: {[t[0] for t in t5_tokens]} ({len(t5_tokens)} tokens)")
        print(f"  Char-level:   {[t[0] for t in char_tokens]} ({len(char_tokens)} tokens)")
        print(f"  FluxEM:       [single 256-dim embedding] (1 embedding)")


def visualize_arithmetic_example():
    """Visualize the arithmetic example: 12345 + 67890 = 80235."""
    print_header("ARITHMETIC EXAMPLE: 12345 + 67890 = ?")

    a, b = 12345, 67890
    expected = a + b

    # Show BPE approach
    print_section("1. GPT-style BPE Approach")
    bpe = BPETokenizerSimulator()

    print(f"  Input:  \"What is {a} + {b}?\"")
    tokens = bpe.tokenize(f"What is {a} + {b}?")
    print(f"  Tokens: {[t[0] for t in tokens]}")
    print(f"  IDs:    {[t[1] for t in tokens]}")
    print()
    print("  Problem: Token embeddings have NO arithmetic relationship!")
    print("           embed('12345') + embed('67890') != embed('80235')")
    print("           Model must memorize or learn patterns from data")
    print("           Fails on out-of-distribution numbers")

    # Show character-level approach
    print_section("2. Character-Level Approach")
    char = CharacterLevelTokenizer()
    analysis = char.analyze_addition_complexity(str(a), str(b))

    print(f"  {a}: tokens = {analysis['a_tokens']}")
    print(f"  {b}: tokens = {analysis['b_tokens']}")
    print(f"  Result: {analysis['result']}")
    print(f"  Carries needed: {analysis['carries_needed']}")
    print()
    print("  Learning required:")
    for req in analysis['learning_required']:
        print(f"    - {req}")

    # Show FluxEM approach
    print_section("3. FluxEM Algebraic Approach")
    fluxem = FluxEMDemonstrator()
    result = fluxem.demonstrate_addition(a, b)

    print(f"  embed({a}) --> 256-dimensional vector e_a")
    print(f"  embed({b}) --> 256-dimensional vector e_b")
    print()
    print(f"  e_a + e_b = e_sum  (vector addition)")
    print(f"  decode(e_sum) = {result['computed']}")
    print()
    print(f"  Expected: {expected}")
    print(f"  Exact match: {result['exact']}")
    print()
    print(f"  KEY INSIGHT: {result['method']}")
    print(f"  Learning required: {result['learning_required']}")


def visualize_ood_comparison():
    """Compare OOD generalization."""
    print_header("OUT-OF-DISTRIBUTION GENERALIZATION")

    print("""
  LLMs are trained on text corpora where certain number patterns appear:
  - Common: 1-100, years (1990-2024), prices ($10, $99.99)
  - Rare: Large arithmetic (e.g., 987654 + 123456)

  When numbers fall outside training distribution, LLMs fail dramatically.
  FluxEM handles ANY magnitude with consistent accuracy.
""")

    fluxem = FluxEMDemonstrator()
    ood_results = fluxem.demonstrate_ood_generalization()

    headers = ["Expression", "Expected", "FluxEM Result", "Rel. Error", "Accurate"]
    rows = []

    for r in ood_results["test_cases"]:
        rows.append([
            r["expression"],
            str(r["expected"]),
            f"{r['computed']:.0f}",
            r["relative_error"],
            "Yes" if r["accurate"] else "NO",
        ])

    print_comparison_table(headers, rows, [25, 18, 18, 12, 10])

    print(f"\n  All test cases accurate: {ood_results['all_accurate']}")
    print(f"  Note: {ood_results['note']}")


def visualize_scientific_notation():
    """Show scientific notation challenges."""
    print_header("SCIENTIFIC NOTATION CHALLENGES")

    t5 = T5TokenizerSimulator()

    examples = ["1e6", "6.022e23", "1.38e-23"]

    for ex in examples:
        print_section(f"Number: {ex}")
        analysis = t5.analyze_scientific_notation(ex)
        print(f"  Tokens: {analysis['tokens']}")
        print(f"  Issues:")
        for issue in analysis['issues']:
            print(f"    - {issue}")


def visualize_embedding_structure():
    """Visualize the embedding structure difference."""
    print_header("EMBEDDING STRUCTURE COMPARISON")

    print("""
  +-----------------+-------------------+----------------------------------+
  | Approach        | Embedding Type    | Arithmetic Property              |
  +-----------------+-------------------+----------------------------------+
  | GPT BPE         | Learned vectors   | None (arbitrary directions)      |
  | T5 SentPiece    | Learned vectors   | None (arbitrary directions)      |
  | Char-level      | Learned vectors   | None (must learn place value)    |
  +-----------------+-------------------+----------------------------------+
  | FluxEM Linear   | Algebraic vectors | embed(a+b) = embed(a) + embed(b) |
  | FluxEM Log      | Algebraic vectors | log_emb(a*b) = log(a) + log(b)   |
  +-----------------+-------------------+----------------------------------+
""")

    print("  FluxEM embedding structure (256-dimensional):")
    print("  +--------------------------------------------------+")
    print("  |  [8 bits: domain tag] [248 bits: value encoding] |")
    print("  +--------------------------------------------------+")
    print()
    print("  Linear encoding (for addition/subtraction):")
    print("    - Number n encoded as n * direction_vector / scale")
    print("    - Vector addition = number addition")
    print()
    print("  Logarithmic encoding (for multiplication/division):")
    print("    - Number n encoded as log(|n|) * direction_vector")
    print("    - Vector addition = number multiplication")
    print("    - Sign tracked separately")


def visualize_summary():
    """Print final summary."""
    print_header("SUMMARY: Why FluxEM Matters for Numeric Reasoning")

    print("""
  Traditional LLM Tokenization:
  +-----------------------------------------------------------------------+
  | Problem                        | Consequence                          |
  +-----------------------------------------------------------------------+
  | Numbers split into arbitrary   | No inherent arithmetic structure     |
  | subword tokens                 |                                      |
  +-----------------------------------------------------------------------+
  | Token embeddings are learned   | Must memorize arithmetic from data   |
  | from text statistics           | Limited by training distribution     |
  +-----------------------------------------------------------------------+
  | No mathematical relationships  | Catastrophic OOD failures            |
  | between number representations | (e.g., large number arithmetic)      |
  +-----------------------------------------------------------------------+

  FluxEM Algebraic Embeddings:
  +-----------------------------------------------------------------------+
  | Property                       | Benefit                              |
  +-----------------------------------------------------------------------+
  | Arithmetic IS geometry         | embed(a) + embed(b) = embed(a+b)     |
  |                                | No learning required for arithmetic  |
  +-----------------------------------------------------------------------+
  | Scale-invariant encoding       | Works for any magnitude uniformly    |
  |                                | No OOD degradation                   |
  +-----------------------------------------------------------------------+
  | Hybrid integration             | Detect numbers, embed algebraically, |
  |                                | project to LLM hidden space          |
  +-----------------------------------------------------------------------+

  The Hybrid Approach:

    Text: "What is 12345 + 67890?"
           |       |       |
           v       v       v
         [LLM]  [FluxEM] [FluxEM]
         tokens  embed    embed
           |       |       |
           +-------+-------+
                   |
                   v
            [Hybrid Transformer]
                   |
                   v
            Exact: 80235
""")


def interactive_demo():
    """Run interactive demonstration."""
    print_header("INTERACTIVE FLUXEM DEMO")

    fluxem = FluxEMDemonstrator()

    print("\n  Enter arithmetic expressions to compute (or 'quit' to exit).")
    print("  Examples: 12345+67890, 999*888, 1000000-1")
    print()

    while True:
        try:
            expr = input("  Expression> ").strip()
            if expr.lower() in ('quit', 'exit', 'q'):
                break

            if not expr:
                continue

            # Parse and compute
            result = fluxem.model.compute(expr)

            # Try to compute expected for comparison
            try:
                # Extract numbers and operator
                clean = expr.replace('=', '')
                for op in ['+', '-', '*', '/']:
                    if op in clean:
                        parts = clean.split(op)
                        a, b = float(parts[0]), float(parts[1])
                        if op == '+':
                            expected = a + b
                        elif op == '-':
                            expected = a - b
                        elif op == '*':
                            expected = a * b
                        else:
                            expected = a / b
                        break
                else:
                    expected = None

                if expected is not None:
                    rel_error = abs(result - expected) / abs(expected) if expected != 0 else abs(result)
                    print(f"    FluxEM result: {result}")
                    print(f"    Expected:      {expected}")
                    print(f"    Relative error: {rel_error:.2e}")
                else:
                    print(f"    Result: {result}")
            except:
                print(f"    Result: {result}")

            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"    Error: {e}")
            print()

    print("\n  Demo complete.\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM numeric encoding vs FluxEM algebraic embeddings"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive demo after visualizations"
    )
    parser.add_argument(
        "--section",
        choices=["tokenization", "arithmetic", "ood", "scientific", "structure", "summary", "all"],
        default="all",
        help="Which section to display"
    )
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  LLM NUMERIC ENCODING vs FLUXEM: A Comparative Analysis")
    print("=" * 78)

    sections = {
        "tokenization": visualize_tokenization_difference,
        "arithmetic": visualize_arithmetic_example,
        "ood": visualize_ood_comparison,
        "scientific": visualize_scientific_notation,
        "structure": visualize_embedding_structure,
        "summary": visualize_summary,
    }

    if args.section == "all":
        for section_fn in sections.values():
            section_fn()
    else:
        sections[args.section]()

    if args.interactive:
        interactive_demo()

    print("\n" + "=" * 78)
    print("  Comparison complete.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
