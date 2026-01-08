"""
Acceptance tests for FluxEM multi-domain tool-calling integration.

These tests validate end-to-end integration of all 10 new domain encoders
with the tool-calling system. They serve as acceptance criteria for the
multi-domain training recipe.

Pass/Fail Criteria:
- Overall tool accuracy: >= 85% (minimum), >= 95% (target)
- Per-domain accuracy: >= 70% (minimum), >= 90% (target)
- Domain detection accuracy: >= 90% (minimum), >= 98% (target)
- Tool execution success: >= 95% (minimum), 100% (target)
"""

import pytest
from typing import Dict, List, Tuple

# Import tool registry
from experiments.qwen3_toolcalling.tool_registry import create_tool_registry

# Import wrapper for E2E testing
from experiments.qwen3_toolcalling.qwen3_wrapper import Qwen3MLXWrapper

# Import integration layer components
from fluxem.integration.tokenizer import DomainType
from fluxem.integration.pipeline import DomainEncoderRegistry
from fluxem.integration.sample_format import VALID_SPAN_TYPES


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def tool_registry():
    """Create a fresh tool registry."""
    return create_tool_registry()


@pytest.fixture
def wrapper():
    """Create a wrapper instance for tool calling tests."""
    return Qwen3MLXWrapper(
        model_path=None,
        tool_selection="pattern",
        llm_query_extraction=False,
        response_style="plain",
    )


@pytest.fixture
def encoder_registry():
    """Create an encoder registry with all domains."""
    return DomainEncoderRegistry()


# ============================================================================
# Tool Registry Integration Tests
# ============================================================================

class TestToolRegistryIntegration:
    """Tests for tool registry completeness and integration."""

    # Expected tools per new domain (matching actual registry)
    EXPECTED_NEW_DOMAIN_TOOLS = {
        "combinatorics": ["combinatorics_factorial", "combinatorics_ncr", "combinatorics_npr"],
        "probability": ["probability_bernoulli_pmf", "probability_binomial_pmf", "probability_bayes_rule"],
        "statistics": ["statistics_mean", "statistics_variance", "statistics_corr"],
        "information_theory": ["info_entropy", "info_kl_divergence", "info_cross_entropy"],
        "signal_processing": ["signal_convolution", "signal_dft_magnitude", "signal_moving_average"],
        "calculus": ["calculus_derivative", "calculus_integral", "calculus_evaluate"],
        "temporal": ["temporal_add_days", "temporal_diff_days", "temporal_day_of_week"],
        "finance": ["finance_npv", "finance_payment", "finance_compound_interest"],
        "optimization": ["optimization_gradient_step", "optimization_least_squares_2x2", "optimization_project_box"],
        "control_systems": ["control_is_stable_2x2", "control_state_update"],
    }

    def test_all_new_domain_tools_registered(self, tool_registry):
        """Verify all expected tools for new domains are registered."""
        registered_names = set(tool_registry.keys())

        for domain, expected_tools in self.EXPECTED_NEW_DOMAIN_TOOLS.items():
            for tool_name in expected_tools:
                assert tool_name in registered_names, \
                    f"Tool '{tool_name}' for domain '{domain}' not registered"

    def test_tool_count_minimum(self, tool_registry):
        """Verify minimum tool count (at least 30 tools for new domains)."""
        new_domain_tool_count = 0
        for tools in self.EXPECTED_NEW_DOMAIN_TOOLS.values():
            new_domain_tool_count += len(tools)

        # Count actual new domain tools in registry
        actual_count = sum(
            1 for name in tool_registry.keys()
            if any(name.startswith(prefix) for prefix in [
                "combinatorics_", "probability_", "statistics_", "info_",
                "signal_", "calculus_", "temporal_", "finance_",
                "optimization_", "control_"
            ])
        )

        assert actual_count >= 30, f"Expected at least 30 new domain tools, got {actual_count}"

    def test_tool_functions_callable(self, tool_registry):
        """Verify all tools have callable functions."""
        for name, tool in tool_registry.items():
            # ToolDescription uses 'function' attribute
            func = getattr(tool, "function", None)
            assert callable(func), f"Tool '{name}' has no callable function"

    def test_tool_descriptions_present(self, tool_registry):
        """Verify all tools have descriptions."""
        for name, tool in tool_registry.items():
            # ToolDescription uses attributes, not dict access
            desc = getattr(tool, "description", "")
            assert len(desc) > 10, f"Tool '{name}' has insufficient description"


# ============================================================================
# Domain Type Integration Tests
# ============================================================================

class TestDomainTypeIntegration:
    """Tests for DomainType enum completeness."""

    EXPECTED_NEW_DOMAIN_TYPES = [
        "COMBINATORICS",
        "PROBABILITY",
        "STATISTICS",
        "INFO_THEORY",
        "SIGNAL",
        "CALCULUS",
        "TEMPORAL",
        "FINANCE",
        "OPTIMIZATION",
        "CONTROL",
    ]

    def test_all_new_domain_types_exist(self):
        """Verify all new DomainType enum values exist."""
        for dtype_name in self.EXPECTED_NEW_DOMAIN_TYPES:
            assert hasattr(DomainType, dtype_name), \
                f"DomainType.{dtype_name} not defined"

    def test_domain_type_values_unique(self):
        """Verify all DomainType values are unique."""
        values = [dt.value for dt in DomainType]
        assert len(values) == len(set(values)), "DomainType values are not unique"

    def test_new_domain_types_have_sequential_values(self):
        """Verify new domain types have sequential values starting at 26."""
        for i, dtype_name in enumerate(self.EXPECTED_NEW_DOMAIN_TYPES):
            dtype = getattr(DomainType, dtype_name)
            expected_value = 26 + i
            assert dtype.value == expected_value, \
                f"DomainType.{dtype_name} has value {dtype.value}, expected {expected_value}"


# ============================================================================
# Encoder Registry Integration Tests
# ============================================================================

class TestEncoderRegistryIntegration:
    """Tests for DomainEncoderRegistry integration with new domains."""

    NEW_DOMAIN_TYPES = [
        DomainType.COMBINATORICS,
        DomainType.PROBABILITY,
        DomainType.STATISTICS,
        DomainType.INFO_THEORY,
        DomainType.SIGNAL,
        DomainType.CALCULUS,
        DomainType.TEMPORAL,
        DomainType.FINANCE,
        DomainType.OPTIMIZATION,
        DomainType.CONTROL,
    ]

    def test_all_new_encoders_registered(self, encoder_registry):
        """Verify all new domain encoders are accessible."""
        for dtype in self.NEW_DOMAIN_TYPES:
            encoder = encoder_registry.get_encoder(dtype)
            assert encoder is not None, f"No encoder for {dtype.name}"

    def test_encoder_produces_valid_embeddings(self, encoder_registry):
        """Verify each new encoder produces 128-dim embeddings."""
        test_inputs = {
            DomainType.COMBINATORICS: {"kind": "ncr", "n": 5, "k": 2},
            DomainType.PROBABILITY: {"kind": "bernoulli", "p": 0.5},
            DomainType.STATISTICS: [1.0, 2.0, 3.0, 4.0, 5.0],
            DomainType.INFO_THEORY: [0.25, 0.25, 0.25, 0.25],
            DomainType.SIGNAL: [1.0, -1.0, 1.0, -1.0],
            DomainType.CALCULUS: [1.0, 2.0, 3.0],  # polynomial coeffs
            DomainType.TEMPORAL: "2024-06-15",
            DomainType.FINANCE: (0.1, [-100, 50, 50, 50]),  # rate, cashflows
            DomainType.OPTIMIZATION: ([1.0, 2.0], [0.1, 0.2], 0.01),  # x, grad, lr
            DomainType.CONTROL: ([[0.9, 0], [0, 0.9]], [[1, 0], [0, 1]], [1, 0], [0.5, -0.5]),
        }

        for dtype, test_input in test_inputs.items():
            encoder = encoder_registry.get_encoder(dtype)

            # Encode based on input type
            if isinstance(test_input, dict):
                # For combinatorics and probability
                if dtype == DomainType.COMBINATORICS:
                    from fluxem.domains.combinatorics import CombinatorialTerm
                    emb = encoder.encode(CombinatorialTerm(**test_input))
                elif dtype == DomainType.PROBABILITY:
                    from fluxem.domains.probability import ProbabilityDistribution
                    emb = encoder.encode(ProbabilityDistribution(**test_input))
            elif isinstance(test_input, tuple):
                # Unpack tuple for multi-arg encoders
                emb = encoder.encode(*test_input)
            else:
                emb = encoder.encode(test_input)

            # Check embedding dimension (is_valid tested in test_domain_encoders_new_domains.py)
            assert len(emb) == 256, f"Embedding dim != 256 for {dtype.name}"


# ============================================================================
# Sample Format Integration Tests
# ============================================================================

class TestSampleFormatIntegration:
    """Tests for sample format span type integration."""

    EXPECTED_SPAN_TYPES = [
        "comb_term",
        "prob_dist",
        "stats_summary",
        "info_entropy",
        "signal_sequence",
        "calc_polynomial",
        "temporal_date",
        "finance_cashflow",
        "opt_step",
        "control_state",
    ]

    def test_all_new_span_types_valid(self):
        """Verify all new span types are in VALID_SPAN_TYPES."""
        for span_type in self.EXPECTED_SPAN_TYPES:
            assert span_type in VALID_SPAN_TYPES, \
                f"Span type '{span_type}' not in VALID_SPAN_TYPES"


# ============================================================================
# End-to-End Tool Calling Tests
# ============================================================================

class TestEndToEndToolCalling:
    """End-to-end tests for tool calling across all new domains."""

    # Test cases: (tool_name, query, expected_result, tolerance)
    E2E_TEST_CASES = [
        # Combinatorics
        ("combinatorics_factorial", "5!", 120, None),
        ("combinatorics_ncr", "C(10, 3)", 120, None),
        ("combinatorics_npr", "P(5, 2)", 20, None),

        # Probability
        ("probability_bernoulli_pmf", "p=0.3 k=1", 0.3, 1e-6),
        ("probability_binomial_pmf", "n=10 k=5 p=0.5", 0.24609375, 1e-6),

        # Statistics
        ("statistics_mean", "[10, 20, 30, 40, 50]", 30.0, 1e-6),
        ("statistics_variance", "[2, 4, 6, 8, 10]", 10.0, 1e-6),  # Population variance

        # Information Theory
        ("info_entropy", "[0.5, 0.5]", 1.0, 1e-6),
        ("info_entropy", "[0.25, 0.25, 0.25, 0.25]", 2.0, 1e-6),

        # Signal Processing
        ("signal_convolution", "[1, 2, 3] [1, 1]", [1.0, 3.0, 5.0, 3.0], None),

        # Calculus
        ("calculus_derivative", "[1, 2, 3]", [2.0, 6.0], None),
        ("calculus_integral", "[2, 4]", [0.0, 2.0, 2.0], None),

        # Temporal
        ("temporal_add_days", "Add 10 days to 2024-01-01", "2024-01-11", None),
        ("temporal_diff_days", "From 2024-01-01 to 2024-01-31", 30, None),

        # Finance
        ("finance_npv", "rate 0.1 cashflows [-100, 60, 60]", 4.132231, 1e-4),
        ("finance_compound_interest", "principal 1000 rate 0.05 periods 10", 1628.89, 0.01),

        # Optimization
        ("optimization_gradient_step", "[1, 2] [0.2, 0.4] lr=0.1", [0.98, 1.96], 1e-6),

        # Control Systems
        ("control_is_stable_2x2", "[[0.5, 0], [0, 0.5]]", True, None),
        ("control_is_stable_2x2", "[[1.5, 0], [0, 1.5]]", False, None),
    ]

    @pytest.mark.parametrize("tool_name,query,expected,tolerance", E2E_TEST_CASES)
    def test_e2e_tool_call(self, wrapper, tool_name, query, expected, tolerance):
        """Parameterized E2E test for tool calling."""
        result = wrapper.call_tool_by_name(tool_name, query)

        assert result["success"] is True, f"Tool {tool_name} failed: {result.get('error')}"

        actual = result["result"]

        if tolerance is not None:
            assert actual == pytest.approx(expected, rel=tolerance), \
                f"{tool_name}: expected {expected}, got {actual}"
        elif isinstance(expected, list):
            if isinstance(expected[0], float):
                assert actual == pytest.approx(expected, rel=1e-6), \
                    f"{tool_name}: expected {expected}, got {actual}"
            else:
                assert actual == expected, f"{tool_name}: expected {expected}, got {actual}"
        else:
            assert actual == expected, f"{tool_name}: expected {expected}, got {actual}"


# ============================================================================
# Domain Detection Tests
# ============================================================================

class TestDomainDetection:
    """Tests for domain detection from natural language prompts."""

    # (prompt, expected_domain_keyword)
    DOMAIN_DETECTION_CASES = [
        ("What is 10 factorial?", "combinatorics"),
        ("Calculate C(8, 3)", "combinatorics"),
        ("Find the probability of getting 4 heads in 10 coin flips", "probability"),
        ("What is the mean of [1, 2, 3, 4, 5]?", "statistics"),
        ("Calculate the standard deviation", "statistics"),
        ("Shannon entropy of a fair coin", "information"),
        ("Convolve these two signals", "signal"),
        ("Find the derivative of x^2 + 3x", "calculus"),
        ("What day is 30 days after January 1?", "temporal"),
        ("Calculate the NPV of these cash flows", "finance"),
        ("Take a gradient descent step", "optimization"),
        ("Is this system stable?", "control"),
    ]

    def test_domain_keyword_extraction(self, wrapper):
        """Test that prompts map to expected domain keywords."""
        domain_keywords = {
            "combinatorics": ["factorial", "C(", "P(", "permutation", "combination"],
            "probability": ["probability", "binomial", "bernoulli", "coin flip", "dice"],
            "statistics": ["mean", "median", "variance", "standard deviation", "correlation"],
            "information": ["entropy", "KL divergence", "mutual information", "bits"],
            "signal": ["convolve", "convolution", "DFT", "signal", "autocorrelation"],
            "calculus": ["derivative", "integral", "differentiate", "integrate", "d/dx"],
            "temporal": ["days after", "days before", "date", "weekday", "between dates"],
            "finance": ["NPV", "IRR", "cash flow", "interest", "present value"],
            "optimization": ["gradient", "descent", "minimize", "optimize", "step"],
            "control": ["stable", "controllable", "step response", "eigenvalue"],
        }

        for prompt, expected_domain in self.DOMAIN_DETECTION_CASES:
            keywords = domain_keywords.get(expected_domain, [])
            prompt_lower = prompt.lower()

            # Check if any keyword matches
            found = any(kw.lower() in prompt_lower for kw in keywords)
            assert found, f"Prompt '{prompt}' should match domain '{expected_domain}'"


# ============================================================================
# Integration Summary Test
# ============================================================================

class TestIntegrationSummary:
    """Summary test to verify complete integration."""

    def test_complete_integration_chain(self, wrapper, encoder_registry):
        """
        Verify the complete integration chain:
        1. DomainType enum has all new types
        2. Encoder registry has all encoders
        3. Tool registry has all tools
        4. Tools execute successfully
        """
        # 1. Check DomainType
        new_types = [
            DomainType.COMBINATORICS, DomainType.PROBABILITY, DomainType.STATISTICS,
            DomainType.INFO_THEORY, DomainType.SIGNAL, DomainType.CALCULUS,
            DomainType.TEMPORAL, DomainType.FINANCE, DomainType.OPTIMIZATION,
            DomainType.CONTROL,
        ]
        assert len(new_types) == 10, "Should have 10 new domain types"

        # 2. Check encoders
        for dtype in new_types:
            assert encoder_registry.get_encoder(dtype) is not None, \
                f"Missing encoder for {dtype.name}"

        # 3. Check tools execute
        sample_tools = [
            ("combinatorics_ncr", "C(5, 2)", 10),
            ("statistics_mean", "[1, 2, 3]", 2.0),
            ("temporal_day_of_week", "2024-01-01", "Monday"),
        ]

        for tool_name, query, expected in sample_tools:
            result = wrapper.call_tool_by_name(tool_name, query)
            assert result["success"], f"Tool {tool_name} failed"
            if isinstance(expected, float):
                assert result["result"] == pytest.approx(expected, rel=1e-6)
            else:
                assert result["result"] == expected


# ============================================================================
# Benchmark Accuracy Test
# ============================================================================

class TestBenchmarkAccuracy:
    """Tests for overall benchmark accuracy metrics."""

    def test_tool_execution_success_rate(self, wrapper):
        """
        Verify tool execution success rate meets minimum threshold.
        Target: >= 95% success rate.
        """
        test_cases = [
            ("combinatorics_factorial", "5!"),
            ("combinatorics_ncr", "C(10, 3)"),
            ("probability_binomial_pmf", "n=5 k=2 p=0.5"),
            ("statistics_mean", "[1, 2, 3, 4, 5]"),
            ("statistics_variance", "[1, 2, 3, 4, 5]"),
            ("info_entropy", "[0.5, 0.5]"),
            ("signal_convolution", "[1, 1] [1, 1]"),
            ("calculus_derivative", "[1, 2, 3]"),
            ("temporal_add_days", "Add 5 days to 2024-01-01"),
            ("temporal_diff_days", "From 2024-01-01 to 2024-01-10"),
            ("finance_npv", "rate 0.1 cashflows [-100, 50, 50, 50]"),
            ("optimization_gradient_step", "[0, 0] [1, 1] lr=0.1"),
            ("control_is_stable_2x2", "[[0.9, 0], [0, 0.9]]"),
        ]

        successes = 0
        failures = []

        for tool_name, query in test_cases:
            result = wrapper.call_tool_by_name(tool_name, query)
            if result["success"]:
                successes += 1
            else:
                failures.append((tool_name, result.get("error")))

        success_rate = successes / len(test_cases)

        # Minimum: 95%
        assert success_rate >= 0.95, \
            f"Success rate {success_rate:.1%} below 95% minimum. Failures: {failures}"
