"""Tests for span detection module."""

import pytest
from fluxem.detection import SpanDetector, DetectedSpan, DOMAIN_PATTERNS
from fluxem.detection.patterns import DomainType
from fluxem.detection.span_detector import (
    SpanDetectorConfig,
    detect_spans,
    detect_numbers,
    detect_combinatorics,
    detect_dates,
    detect_formulas,
    detect_quantities,
)


class TestDetectedSpan:
    """Tests for DetectedSpan dataclass."""

    def test_basic_creation(self):
        span = DetectedSpan(
            start=0,
            end=5,
            text="hello",
            domain=DomainType.ARITHMETIC,
            confidence=0.9,
            parsed_value=123,
        )
        assert span.start == 0
        assert span.end == 5
        assert span.text == "hello"
        assert span.length == 5

    def test_overlaps_true(self):
        span1 = DetectedSpan(0, 10, "abc", DomainType.ARITHMETIC, 0.9, None)
        span2 = DetectedSpan(5, 15, "def", DomainType.ARITHMETIC, 0.9, None)
        assert span1.overlaps(span2)
        assert span2.overlaps(span1)

    def test_overlaps_false(self):
        span1 = DetectedSpan(0, 5, "abc", DomainType.ARITHMETIC, 0.9, None)
        span2 = DetectedSpan(10, 15, "def", DomainType.ARITHMETIC, 0.9, None)
        assert not span1.overlaps(span2)
        assert not span2.overlaps(span1)

    def test_overlaps_adjacent(self):
        span1 = DetectedSpan(0, 5, "abc", DomainType.ARITHMETIC, 0.9, None)
        span2 = DetectedSpan(5, 10, "def", DomainType.ARITHMETIC, 0.9, None)
        # Adjacent spans should not overlap
        assert not span1.overlaps(span2)

    def test_contains(self):
        span1 = DetectedSpan(0, 20, "abc", DomainType.ARITHMETIC, 0.9, None)
        span2 = DetectedSpan(5, 15, "def", DomainType.ARITHMETIC, 0.9, None)
        assert span1.contains(span2)
        assert not span2.contains(span1)


class TestSpanDetectorArithmetic:
    """Tests for arithmetic span detection."""

    def test_detect_integer(self):
        spans = detect_numbers("The answer is 42")
        assert len(spans) == 1
        assert spans[0].text == "42"
        assert spans[0].parsed_value == 42

    def test_detect_negative_integer(self):
        spans = detect_numbers("Temperature is -10 degrees")
        assert any(s.text == "-10" for s in spans)

    def test_detect_float(self):
        spans = detect_numbers("Pi is approximately 3.14159")
        assert any("3.14159" in s.text for s in spans)

    def test_detect_scientific_notation(self):
        spans = detect_numbers("Speed of light is 3e8 m/s")
        assert any("3e8" in s.text for s in spans)

    def test_detect_comma_separated(self):
        spans = detect_numbers("Population: 1,000,000")
        # Should recognize the number with commas
        assert len(spans) >= 1


class TestSpanDetectorCombinatorics:
    """Tests for combinatorics span detection."""

    def test_detect_factorial(self):
        spans = detect_combinatorics("Calculate 5!")
        assert len(spans) == 1
        assert spans[0].text == "5!"
        assert spans[0].parsed_value["type"] == "factorial"
        assert spans[0].parsed_value["n"] == 5

    def test_detect_double_factorial(self):
        spans = detect_combinatorics("What is 7!!?")
        assert len(spans) == 1
        assert spans[0].text == "7!!"
        assert spans[0].parsed_value["type"] == "double_factorial"

    def test_detect_combination_C_notation(self):
        spans = detect_combinatorics("C(10, 3) = ?")
        assert len(spans) == 1
        assert spans[0].parsed_value["type"] == "combination"
        assert spans[0].parsed_value["n"] == 10
        assert spans[0].parsed_value["k"] == 3

    def test_detect_combination_choose(self):
        spans = detect_combinatorics("5 choose 2")
        assert len(spans) == 1
        assert spans[0].parsed_value["type"] == "combination"

    def test_detect_permutation(self):
        spans = detect_combinatorics("P(5, 3)")
        assert len(spans) == 1
        assert spans[0].parsed_value["type"] == "permutation"


class TestSpanDetectorProbability:
    """Tests for probability span detection."""

    def test_detect_bernoulli(self):
        detector = SpanDetector(
            SpanDetectorConfig(domains={DomainType.PROBABILITY})
        )
        spans = detector.detect("Bernoulli(0.5)")
        assert len(spans) == 1
        assert spans[0].parsed_value["kind"] == "bernoulli"
        assert spans[0].parsed_value["p"] == 0.5

    def test_detect_binomial(self):
        detector = SpanDetector(
            SpanDetectorConfig(domains={DomainType.PROBABILITY})
        )
        spans = detector.detect("Binomial(10, 0.3)")
        assert len(spans) == 1
        assert spans[0].parsed_value["kind"] == "binomial"
        assert spans[0].parsed_value["n"] == 10


class TestSpanDetectorTemporal:
    """Tests for temporal span detection."""

    def test_detect_iso_date(self):
        spans = detect_dates("Meeting on 2024-01-15")
        assert len(spans) == 1
        assert spans[0].text == "2024-01-15"
        assert spans[0].parsed_value["year"] == 2024
        assert spans[0].parsed_value["month"] == 1
        assert spans[0].parsed_value["day"] == 15

    def test_detect_us_date(self):
        spans = detect_dates("Due: 12/25/2024")
        assert len(spans) == 1
        assert spans[0].parsed_value["month"] == 12
        assert spans[0].parsed_value["day"] == 25


class TestSpanDetectorChemistry:
    """Tests for chemistry span detection."""

    def test_detect_water(self):
        spans = detect_formulas("H2O is water")
        assert len(spans) >= 1
        # Find the H2O span
        h2o_spans = [s for s in spans if "H" in s.text and "O" in s.text]
        assert len(h2o_spans) >= 1

    def test_detect_glucose(self):
        spans = detect_formulas("Glucose is C6H12O6")
        h_spans = [s for s in spans if "C" in s.text and "H" in s.text]
        assert len(h_spans) >= 1


class TestSpanDetectorPhysics:
    """Tests for physics span detection."""

    def test_detect_velocity(self):
        spans = detect_quantities("Speed is 100 m/s")
        assert len(spans) >= 1

    def test_detect_mass(self):
        spans = detect_quantities("Mass: 5.5 kg")
        assert len(spans) >= 1


class TestSpanDetectorMultipleDomains:
    """Tests for multi-domain detection."""

    def test_mixed_domains(self):
        detector = SpanDetector()
        text = "Calculate 5! + 42 on 2024-01-15"
        spans = detector.detect(text)

        domains = {s.domain for s in spans}
        # Should detect combinatorics (5!), arithmetic (42), and temporal (date)
        assert DomainType.COMBINATORICS in domains
        assert DomainType.TEMPORAL in domains

    def test_overlap_resolution_priority(self):
        detector = SpanDetector(
            SpanDetectorConfig(resolve_overlaps="priority")
        )
        # "5!" could match as both factorial and number
        spans = detector.detect("5!")
        # Higher priority (factorial) should win
        assert len(spans) == 1
        assert spans[0].domain == DomainType.COMBINATORICS


class TestSpanDetectorConfig:
    """Tests for detector configuration."""

    def test_domain_filter(self):
        detector = SpanDetector(
            SpanDetectorConfig(domains={DomainType.ARITHMETIC})
        )
        spans = detector.detect("Calculate 5! + 42")
        # Should only detect arithmetic, not combinatorics
        for span in spans:
            assert span.domain == DomainType.ARITHMETIC

    def test_min_confidence(self):
        detector = SpanDetector(SpanDetectorConfig(min_confidence=0.99))
        spans = detector.detect("42")
        # High confidence threshold may filter some matches
        # This depends on pattern priorities

    def test_max_spans(self):
        detector = SpanDetector(SpanDetectorConfig(max_spans=1))
        spans = detector.detect("1 2 3 4 5")
        assert len(spans) <= 1


class TestSpanDetectorUtilities:
    """Tests for utility methods."""

    def test_get_non_span_regions(self):
        detector = SpanDetector()
        text = "abc 42 def 5! xyz"
        spans = detector.detect(text)
        regions = detector.get_non_span_regions(text, spans)

        # Check that regions cover non-matched parts
        for start, end in regions:
            region_text = text[start:end]
            # Regions should not contain our detected values
            assert "42" not in region_text or "5!" not in region_text

    def test_annotate(self):
        detector = SpanDetector(
            SpanDetectorConfig(domains={DomainType.COMBINATORICS})
        )
        text = "Calculate 5!"
        annotated = detector.annotate(text)
        assert "[COMBINATORICS:" in annotated


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_spans(self):
        spans = detect_spans("5! + 42")
        assert len(spans) >= 1

    def test_detect_numbers_function(self):
        spans = detect_numbers("100")
        assert len(spans) == 1

    def test_detect_combinatorics_function(self):
        spans = detect_combinatorics("10!")
        assert len(spans) == 1

    def test_detect_dates_function(self):
        spans = detect_dates("2024-01-01")
        assert len(spans) == 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        detector = SpanDetector()
        spans = detector.detect("")
        assert spans == []

    def test_no_matches(self):
        detector = SpanDetector()
        spans = detector.detect("hello world")
        # May have some false positives from chemistry, etc.
        # At minimum, no arithmetic
        arith_spans = [s for s in spans if s.domain == DomainType.ARITHMETIC]
        assert len(arith_spans) == 0

    def test_unicode(self):
        detector = SpanDetector()
        # Should handle unicode gracefully
        spans = detector.detect("∫ x dx")
        # May or may not detect integral depending on patterns

    def test_very_long_text(self):
        detector = SpanDetector()
        text = " ".join(["42"] * 100)
        spans = detector.detect(text)
        # Should handle without error
        assert len(spans) >= 1


class TestPatternRegistry:
    """Tests for pattern registry."""

    def test_all_domains_have_patterns(self):
        for domain in DomainType:
            assert domain in DOMAIN_PATTERNS
            assert len(DOMAIN_PATTERNS[domain]) > 0

    def test_patterns_sorted_by_priority(self):
        from fluxem.detection.patterns import get_all_patterns

        patterns = get_all_patterns()
        priorities = [p.priority for p in patterns]
        assert priorities == sorted(priorities, reverse=True)
