#!/usr/bin/env python3
"""Compare algebraic and learned embeddings on arithmetic examples.

Runs small tests using:
- FluxEM arithmetic embeddings
- Sentence-transformers (optional)
- A character-level baseline
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from fluxem.arithmetic.unified import create_unified_model


# =============================================================================
# Sentence-Transformer Support (Optional)
# =============================================================================

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Optional[Any]:
    """Load sentence-transformer model if available."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"Warning: Could not load sentence-transformer model: {e}")
        return None


# =============================================================================
# Character-Level Baseline
# =============================================================================

def char_level_encode(text: str, dim: int = 256) -> np.ndarray:
    """
    Simple character-level encoding baseline.

    Creates a bag-of-characters embedding normalized by length.
    No arithmetic understanding - just character statistics.
    """
    embedding = np.zeros(dim)

    for char in text:
        # Hash character to a position
        idx = ord(char) % dim
        embedding[idx] += 1

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def char_level_similarity(text1: str, text2: str, dim: int = 256) -> float:
    """Compute cosine similarity between character-level embeddings."""
    emb1 = char_level_encode(text1, dim)
    emb2 = char_level_encode(text2, dim)
    return float(np.dot(emb1, emb2))


# =============================================================================
# FluxEM Arithmetic
# =============================================================================

@dataclass
class FluxEMResult:
    """Result of FluxEM arithmetic computation."""
    operand1: float
    operand2: float
    operator: str
    computed_result: float
    expected_result: float
    is_exact: bool
    relative_error: float


def fluxem_compute_addition(a: float, b: float) -> FluxEMResult:
    """Compute `a + b` using FluxEM embeddings."""
    model = create_unified_model(dim=256, linear_scale=1e12)

    # Encode both numbers
    emb_a = model.linear_encoder.encode_number(a)
    emb_b = model.linear_encoder.encode_number(b)

    # Add embeddings (algebraic operation)
    emb_sum = emb_a + emb_b

    # Decode result
    computed = model.linear_encoder.decode(emb_sum)
    expected = a + b

    # Check exactness
    rel_error = abs(computed - expected) / max(abs(expected), 1e-10)
    is_exact = rel_error < 1e-6

    return FluxEMResult(
        operand1=a,
        operand2=b,
        operator="+",
        computed_result=computed,
        expected_result=expected,
        is_exact=is_exact,
        relative_error=rel_error,
    )


def fluxem_verify_commutativity(a: float, b: float) -> Tuple[bool, float]:
    """Verify `a + b == b + a` via embedding-space computation."""
    model = create_unified_model(dim=256, linear_scale=1e12)

    emb_a = model.linear_encoder.encode_number(a)
    emb_b = model.linear_encoder.encode_number(b)

    # a + b
    result_ab = model.linear_encoder.decode(emb_a + emb_b)

    # b + a
    result_ba = model.linear_encoder.decode(emb_b + emb_a)

    diff = abs(result_ab - result_ba)
    is_commutative = diff < 1e-10

    return is_commutative, diff


# =============================================================================
# Sentence-Transformer Experiments
# =============================================================================

def st_similarity_matrix(model: Any, texts: List[str]) -> np.ndarray:
    """Compute pairwise cosine similarities for a list of texts."""
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Pairwise cosine similarity
    return np.dot(normalized, normalized.T)


def st_can_predict_result(
    model: Any,
    expression: str,
    candidates: List[str],
) -> Tuple[str, float]:
    """
    Test if sentence-transformer can predict arithmetic result via similarity.

    Given an expression like "2 + 2", find which candidate is most similar.
    """
    all_texts = [expression] + candidates
    embeddings = model.encode(all_texts, convert_to_numpy=True)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Similarity of expression to each candidate
    expr_emb = normalized[0]
    candidate_embs = normalized[1:]

    similarities = np.dot(candidate_embs, expr_emb)
    best_idx = np.argmax(similarities)

    return candidates[best_idx], float(similarities[best_idx])


# =============================================================================
# Test Cases
# =============================================================================

@dataclass
class TestCase:
    """A test case for comparing embedding approaches."""
    name: str
    expression: str
    operand1: float
    operand2: float
    expected: float
    description: str


TEST_CASES = [
    # Basic arithmetic
    TestCase(
        name="simple_add",
        expression="2 + 2",
        operand1=2,
        operand2=2,
        expected=4,
        description="Simple addition within training distribution",
    ),
    TestCase(
        name="larger_add",
        expression="123 + 456",
        operand1=123,
        operand2=456,
        expected=579,
        description="Larger numbers, still reasonable",
    ),

    # Out-of-distribution (OOD) - tests generalization
    TestCase(
        name="ood_large",
        expression="1000000 + 1",
        operand1=1000000,
        operand2=1,
        expected=1000001,
        description="OOD: Large magnitude (learned models fail here)",
    ),
    TestCase(
        name="ood_extreme",
        expression="999999999 + 1",
        operand1=999999999,
        operand2=1,
        expected=1000000000,
        description="OOD: Extreme magnitude, carry propagation",
    ),

    # Commutativity tests
    TestCase(
        name="commutative_small",
        expression="3 + 5",
        operand1=3,
        operand2=5,
        expected=8,
        description="Commutativity test: 3 + 5 should equal 5 + 3",
    ),
    TestCase(
        name="commutative_large",
        expression="12345 + 67890",
        operand1=12345,
        operand2=67890,
        expected=80235,
        description="Commutativity with larger numbers",
    ),
]


# =============================================================================
# Comparison Runner
# =============================================================================

def run_fluxem_tests(test_cases: List[TestCase]) -> Dict[str, FluxEMResult]:
    """Run all test cases through FluxEM."""
    results = {}

    for tc in test_cases:
        result = fluxem_compute_addition(tc.operand1, tc.operand2)
        results[tc.name] = result

    return results


def run_sentence_transformer_tests(
    model: Any,
    test_cases: List[TestCase],
) -> Dict[str, Dict[str, Any]]:
    """Run sentence-transformer similarity tests."""
    results = {}

    for tc in test_cases:
        # Test: Can ST identify the correct result via similarity?
        # Provide correct answer and some distractors
        correct = str(int(tc.expected))
        distractors = [
            str(int(tc.expected) + 1),
            str(int(tc.expected) - 1),
            str(int(tc.operand1)),
            str(int(tc.operand2)),
        ]
        candidates = [correct] + distractors

        predicted, similarity = st_can_predict_result(model, tc.expression, candidates)

        # Also test commutativity for applicable cases
        if tc.name.startswith("commutative"):
            reversed_expr = f"{int(tc.operand2)} + {int(tc.operand1)}"

            # Get embeddings
            embeddings = model.encode([tc.expression, reversed_expr], convert_to_numpy=True)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / norms
            comm_similarity = float(np.dot(normalized[0], normalized[1]))
        else:
            comm_similarity = None

        results[tc.name] = {
            "expression": tc.expression,
            "expected": correct,
            "predicted": predicted,
            "similarity": similarity,
            "correct": predicted == correct,
            "commutative_similarity": comm_similarity,
        }

    return results


def run_char_level_tests(test_cases: List[TestCase]) -> Dict[str, Dict[str, Any]]:
    """Run character-level baseline tests."""
    results = {}

    for tc in test_cases:
        # Similarity between expression and result
        expr_result_sim = char_level_similarity(tc.expression, str(int(tc.expected)))

        # Commutativity
        if tc.name.startswith("commutative"):
            reversed_expr = f"{int(tc.operand2)} + {int(tc.operand1)}"
            comm_sim = char_level_similarity(tc.expression, reversed_expr)
        else:
            comm_sim = None

        results[tc.name] = {
            "expression": tc.expression,
            "expected": str(int(tc.expected)),
            "expr_result_similarity": expr_result_sim,
            "commutative_similarity": comm_sim,
        }

    return results


# =============================================================================
# Output Formatting
# =============================================================================

def print_header(title: str):
    """Print a section header."""
    print(f"table={title}")


def print_comparison_table(
    fluxem_results: Dict[str, FluxEMResult],
    st_results: Optional[Dict[str, Dict[str, Any]]],
    char_results: Dict[str, Dict[str, Any]],
    test_cases: List[TestCase],
):
    """Print formatted comparison table."""

    print_header("comparison")
    print("test_case\texpression\texpected\tfluxem_result\tfluxem_exact\tst_pred\tst_correct\tchar_similarity")

    for tc in test_cases:
        fluxem = fluxem_results[tc.name]
        char = char_results[tc.name]

        fluxem_str = f"{fluxem.computed_result:.6f}"
        char_str = f"{char['expr_result_similarity']:.6f}"

        if st_results:
            st = st_results[tc.name]
            st_pred = st['predicted']
            st_correct = "1" if st['correct'] else "0"
        else:
            st_pred = ""
            st_correct = ""

        row = [
            tc.name,
            tc.expression,
            f"{tc.expected:.0f}",
            fluxem_str,
            "1" if fluxem.is_exact else "0",
            st_pred,
            st_correct,
            char_str,
        ]
        print("\t".join(row))


def print_detailed_analysis(
    fluxem_results: Dict[str, FluxEMResult],
    st_results: Optional[Dict[str, Dict[str, Any]]],
    test_cases: List[TestCase],
):
    """Print detailed analysis of results."""

    print_header("fluxem_details")
    print("test_case\texpression\tcomputed_result\trelative_error\tis_exact")

    for tc in test_cases:
        result = fluxem_results[tc.name]
        row = [
            tc.name,
            tc.expression,
            f"{result.computed_result:.6f}",
            f"{result.relative_error:.6e}",
            "1" if result.is_exact else "0",
        ]
        print("\t".join(row))

    print_header("commutativity_check")
    print("operand1\toperand2\tis_commutative\tdiff")
    for tc in test_cases:
        if tc.name.startswith("commutative"):
            is_comm, diff = fluxem_verify_commutativity(tc.operand1, tc.operand2)
            print(f"{tc.operand1}\t{tc.operand2}\t{'1' if is_comm else '0'}\t{diff:.6e}")

    # Sentence-Transformer Analysis
    if st_results:
        print_header("sentence_transformer_details")
        print("test_case\texpression\tpredicted\tcorrect\tsimilarity\tcommutative_similarity")
        for tc in test_cases:
            result = st_results[tc.name]
            comm_sim = result['commutative_similarity']
            comm_sim_str = f"{comm_sim:.6f}" if comm_sim is not None else ""
            row = [
                tc.name,
                tc.expression,
                result['predicted'],
                "1" if result['correct'] else "0",
                f"{result['similarity']:.6f}",
                comm_sim_str,
            ]
            print("\t".join(row))
    else:
        print_header("sentence_transformer_details")
        print("note\tvalue")
        print("availability\tmissing")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the comparison experiment."""

    print_header("run_context")
    print("field\tvalue")
    st_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("sentence_transformers_available\t1")
        st_model = load_sentence_transformer()
        print(f"sentence_transformers_loaded\t{'1' if st_model else '0'}")
    else:
        print("sentence_transformers_available\t0")
        print("sentence_transformers_loaded\t0")

    # Run tests
    fluxem_results = run_fluxem_tests(TEST_CASES)

    st_results = None
    if st_model:
        st_results = run_sentence_transformer_tests(st_model, TEST_CASES)

    char_results = run_char_level_tests(TEST_CASES)

    # Output results
    print_comparison_table(fluxem_results, st_results, char_results, TEST_CASES)
    print_detailed_analysis(fluxem_results, st_results, TEST_CASES)

    # Summary statistics
    print_header("summary")
    print("metric\tvalue")
    fluxem_exact = sum(1 for r in fluxem_results.values() if r.is_exact)
    print(f"fluxem_exact_results\t{fluxem_exact}/{len(TEST_CASES)}")

    if st_results:
        st_correct = sum(1 for r in st_results.values() if r['correct'])
        print(f"st_correct_predictions\t{st_correct}/{len(TEST_CASES)}")

    # OOD specific
    ood_cases = [tc for tc in TEST_CASES if tc.name.startswith("ood")]
    if ood_cases:
        ood_exact = sum(1 for tc in ood_cases if fluxem_results[tc.name].is_exact)
        print(f"ood_fluxem_exact\t{ood_exact}/{len(ood_cases)}")

        if st_results:
            ood_correct = sum(1 for tc in ood_cases if st_results[tc.name]['correct'])
            print(f"ood_st_correct\t{ood_correct}/{len(ood_cases)}")


if __name__ == "__main__":
    main()
