#!/usr/bin/env python3
"""Numeric encoding comparison.

Prints example tokenizations and arithmetic results for several tokenizer
simulators and FluxEM.
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
# FluxEM example
# =============================================================================

class FluxEMExampleRunner:
    """Example FluxEM arithmetic via embedding operations."""

    def __init__(self):
        self.model = create_unified_model(dim=256)
        self.linear_encoder = self.model.linear_encoder
        self.log_encoder = self.model.log_encoder

    def example_addition(self, a: float, b: float) -> Dict[str, Any]:
        """Example addition via embedding arithmetic."""
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
            "learning_required": "No training required for the arithmetic operator",
        }

    def example_multiplication(self, a: float, b: float) -> Dict[str, Any]:
        """Example multiplication via log embedding arithmetic."""
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

    def example_ood_generalization(self) -> Dict[str, Any]:
        """Example out-of-distribution generalization."""
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
    _ = width
    table_name = title.strip().lower().replace(" ", "_")
    print(f"table={table_name}")


def print_section(title: str, width: int = 78):
    """Print a section header."""
    _ = width
    section_name = title.strip().lower().replace(" ", "_")
    print(f"section={section_name}")


def print_comparison_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None):
    """Print a formatted comparison table."""
    _ = col_widths
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(c) for c in row))


def visualize_tokenization_difference():
    """Visualize how different tokenizers handle the same number."""
    print_header("Tokenization")

    test_numbers = ["123456", "67890", "80235"]

    bpe = BPETokenizerSimulator()
    t5 = T5TokenizerSimulator()
    char = CharacterLevelTokenizer()

    print("number\ttokenizer\ttokens\tcount")
    for num in test_numbers:
        bpe_result = bpe.analyze_number(num)
        t5_tokens = t5.tokenize(num)
        char_tokens = char.tokenize(num)

        print(f"{num}\tgpt_bpe\t{bpe_result['tokens']}\t{len(bpe_result['tokens'])}")
        print(f"{num}\tt5_sentencepiece\t{[t[0] for t in t5_tokens]}\t{len(t5_tokens)}")
        print(f"{num}\tchar_level\t{[t[0] for t in char_tokens]}\t{len(char_tokens)}")
        print(f"{num}\tfluxem\t[single_256_dim_embedding]\t1")


def visualize_arithmetic_example():
    """Visualize the arithmetic example: 12345 + 67890 = 80235."""
    print_header("Arithmetic example")

    a, b = 12345, 67890
    expected = a + b

    # Show BPE approach
    bpe = BPETokenizerSimulator()

    tokens = bpe.tokenize(f"What is {a} + {b}?")
    print("table=arithmetic_bpe")
    print("expression\ttokens\tids")
    print(f"What is {a} + {b}?\t{[t[0] for t in tokens]}\t{[t[1] for t in tokens]}")

    # Show character-level approach
    char = CharacterLevelTokenizer()
    analysis = char.analyze_addition_complexity(str(a), str(b))

    print("table=arithmetic_char_analysis")
    print("a\tb\ta_tokens\tb_tokens\tresult\tcarries_needed\tlearning_required")
    learning_required = "|".join(analysis["learning_required"])
    print(f"{a}\t{b}\t{analysis['a_tokens']}\t{analysis['b_tokens']}\t{analysis['result']}\t{analysis['carries_needed']}\t{learning_required}")

    # Show FluxEM approach
    fluxem = FluxEMExampleRunner()
    result = fluxem.example_addition(a, b)

    print("table=arithmetic_fluxem")
    print("a\tb\texpected\tcomputed\texact\tmethod\tlearning_required")
    print(f"{a}\t{b}\t{expected}\t{result['computed']}\t{result['exact']}\t{result['method']}\t{result['learning_required']}")


def visualize_ood_comparison():
    """Compare OOD generalization."""
    print_header("OOD magnitude")

    fluxem = FluxEMExampleRunner()
    ood_results = fluxem.example_ood_generalization()

    headers = ["expression", "expected", "fluxem_result", "relative_error", "accurate"]
    rows = []

    for r in ood_results["test_cases"]:
        rows.append([
            r["expression"],
            str(r["expected"]),
            f"{r['computed']:.0f}",
            r["relative_error"],
            "1" if r["accurate"] else "0",
        ])

    print_comparison_table(headers, rows, [25, 18, 18, 12, 10])

    print("table=ood_summary")
    print("all_accurate")
    print(f"{ood_results['all_accurate']}")


def visualize_scientific_notation():
    """Show scientific notation challenges."""
    print_header("Scientific notation")

    t5 = T5TokenizerSimulator()

    examples = ["1e6", "6.022e23", "1.38e-23"]

    print("number\ttokens\tissues")
    for ex in examples:
        analysis = t5.analyze_scientific_notation(ex)
        issues = "|".join(analysis["issues"])
        print(f"{ex}\t{analysis['tokens']}\t{issues}")


def visualize_embedding_structure():
    """Visualize the embedding structure difference."""
    print_header("Embedding structure")

    print("field\tvalue")
    print("embedding_dim\t256")
    print("domain_tag_bits\t8")
    print("value_encoding_bits\t248")
    print("linear_encoding\tNumber n encoded as n * direction_vector / scale")
    print("linear_operator\tVector addition equals number addition")
    print("log_encoding\tNumber n encoded as log(|n|) * direction_vector")
    print("log_operator\tVector addition equals number multiplication")
    print("sign_tracking\tstored separately")


def visualize_summary():
    """Print final summary."""
    print_header("Summary")
    print("identity\tformula")
    print("linear_addition\tembed(a) + embed(b) = embed(a+b)")
    print("log_multiplication\tlog_embed(a*b) = log_embed(a) + log_embed(b)")


def interactive_demo():
    """Run interactive example."""
    print_header("Interactive")

    fluxem = FluxEMExampleRunner()

    print("note\tvalue")
    print("interactive_prompt\tenter arithmetic expressions or 'quit'")
    print("examples\t12345+67890, 999*888, 1000000-1")
    print("expression\tcomputed\texpected\trelative_error")

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
                    print(f"{expr}\t{result}\t{expected}\t{rel_error:.6e}")
                else:
                    print(f"{expr}\t{result}\t\t")
            except:
                print(f"{expr}\t{result}\t\t")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"error\t{e}")


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

    print_header("Numeric encoding comparison")

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


if __name__ == "__main__":
    main()
