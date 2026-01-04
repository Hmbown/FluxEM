"""
Tests for JSONL sample format validation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from fluxem.integration.sample_format import (
    Sample,
    Span,
    ValidationError,
    ValidationResult,
    validate_sample,
    validate_span,
    validate_jsonl_file,
    VALID_SPAN_TYPES,
)


class TestSpan:
    """Tests for Span dataclass."""
    
    def test_span_to_dict_basic(self):
        span = Span(type="arithmetic", start=0, end=5, value=123.0)
        d = span.to_dict()
        assert d["type"] == "arithmetic"
        assert d["start"] == 0
        assert d["end"] == 5
        assert d["value"] == 123.0
    
    def test_span_to_dict_with_dims(self):
        span = Span(type="phys_quantity", start=0, end=10, value=373.15, dims={"Theta": 1})
        d = span.to_dict()
        assert d["dims"] == {"Theta": 1}
    
    def test_span_from_dict(self):
        d = {"type": "arithmetic", "start": 5, "end": 10, "value": 42.0}
        span = Span.from_dict(d)
        assert span.type == "arithmetic"
        assert span.start == 5
        assert span.end == 10
        assert span.value == 42.0
    
    def test_span_roundtrip(self):
        original = Span(type="math_complex", start=0, end=5, value={"real": 3, "imag": 4})
        d = original.to_dict()
        restored = Span.from_dict(d)
        assert restored.type == original.type
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.value == original.value


class TestSample:
    """Tests for Sample dataclass."""
    
    def test_sample_to_dict_minimal(self):
        sample = Sample(text="1 + 2", target_text="3")
        d = sample.to_dict()
        assert d["text"] == "1 + 2"
        assert d["target_text"] == "3"
        assert "spans" not in d  # Empty spans should be omitted
    
    def test_sample_to_dict_with_spans(self):
        sample = Sample(
            text="1 + 2",
            target_text="3",
            spans=[Span(type="arithmetic", start=0, end=1, value=1.0)]
        )
        d = sample.to_dict()
        assert "spans" in d
        assert len(d["spans"]) == 1
    
    def test_sample_from_dict(self):
        d = {
            "text": "Calculate 100 + 200",
            "target_text": "300",
            "spans": [{"type": "arithmetic", "start": 10, "end": 13, "value": 100.0}],
            "target_value": 300.0,
        }
        sample = Sample.from_dict(d)
        assert sample.text == "Calculate 100 + 200"
        assert sample.target_text == "300"
        assert len(sample.spans) == 1
        assert sample.target_value == 300.0
    
    def test_sample_json_roundtrip(self):
        original = Sample(
            text="H2O at 373 K",
            target_text="boiling",
            spans=[
                Span(type="chem_formula", start=0, end=3, value="H2O"),
                Span(type="phys_quantity", start=7, end=12, value=373.0, dims={"Theta": 1}),
            ]
        )
        json_str = original.to_json()
        restored = Sample.from_json(json_str)
        assert restored.text == original.text
        assert restored.target_text == original.target_text
        assert len(restored.spans) == 2


class TestValidateSpan:
    """Tests for span validation."""
    
    def test_valid_span(self):
        span = {"type": "arithmetic", "start": 0, "end": 5}
        errors = validate_span(span, "12345")
        assert len(errors) == 0
    
    def test_missing_type(self):
        span = {"start": 0, "end": 5}
        errors = validate_span(span, "12345")
        assert any("type" in e.field for e in errors)
    
    def test_invalid_type(self):
        span = {"type": "invalid_type", "start": 0, "end": 5}
        errors = validate_span(span, "12345")
        assert any("Unknown span type" in e.message for e in errors)
    
    def test_missing_start(self):
        span = {"type": "arithmetic", "end": 5}
        errors = validate_span(span, "12345")
        assert any("start" in e.field for e in errors)
    
    def test_negative_start(self):
        span = {"type": "arithmetic", "start": -1, "end": 5}
        errors = validate_span(span, "12345")
        assert any("non-negative" in e.message for e in errors)
    
    def test_start_greater_than_end(self):
        span = {"type": "arithmetic", "start": 10, "end": 5}
        errors = validate_span(span, "12345")
        assert any("less than end" in e.message for e in errors)
    
    def test_end_exceeds_text_length(self):
        span = {"type": "arithmetic", "start": 0, "end": 100}
        errors = validate_span(span, "12345")
        assert any("exceeds text length" in e.message for e in errors)
    
    def test_all_valid_span_types(self):
        """Ensure all documented span types are accepted."""
        for span_type in VALID_SPAN_TYPES:
            span = {"type": span_type, "start": 0, "end": 5}
            errors = validate_span(span, "12345")
            type_errors = [e for e in errors if "Unknown span type" in e.message]
            assert len(type_errors) == 0, f"Type {span_type} should be valid"


class TestValidateSample:
    """Tests for sample validation."""
    
    def test_valid_sample_minimal(self):
        sample = {"text": "1 + 2", "target_text": "3"}
        result = validate_sample(sample)
        assert result.valid
        assert len(result.errors) == 0
    
    def test_valid_sample_with_spans(self):
        sample = {
            "text": "Calculate 100",
            "target_text": "100",
            "spans": [{"type": "arithmetic", "start": 10, "end": 13, "value": 100.0}]
        }
        result = validate_sample(sample)
        assert result.valid
    
    def test_missing_text(self):
        sample = {"target_text": "3"}
        result = validate_sample(sample)
        assert not result.valid
        assert any("text" in e.field for e in result.errors)
    
    def test_missing_target_text(self):
        sample = {"text": "1 + 2"}
        result = validate_sample(sample)
        assert not result.valid
        assert any("target_text" in e.field for e in result.errors)
    
    def test_invalid_spans_type(self):
        sample = {"text": "test", "target_text": "test", "spans": "not a list"}
        result = validate_sample(sample)
        assert not result.valid
        assert any("spans must be a list" in e.message for e in result.errors)
    
    def test_overlapping_spans_warning(self):
        sample = {
            "text": "12345",
            "target_text": "test",
            "spans": [
                {"type": "arithmetic", "start": 0, "end": 3},
                {"type": "arithmetic", "start": 2, "end": 5},
            ]
        }
        result = validate_sample(sample)
        assert result.valid  # Overlaps are warnings, not errors
        assert len(result.warnings) > 0
        assert any("overlap" in w for w in result.warnings)


class TestValidateJsonlFile:
    """Tests for JSONL file validation."""
    
    def test_valid_file(self):
        samples = [
            {"text": "1 + 2", "target_text": "3"},
            {"text": "3 * 4", "target_text": "12"},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            path = f.name
        
        try:
            results = validate_jsonl_file(path)
            assert results["valid"]
            assert results["total_samples"] == 2
            assert results["valid_samples"] == 2
            assert results["invalid_samples"] == 0
        finally:
            Path(path).unlink()
    
    def test_invalid_json_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "valid", "target_text": "ok"}\n')
            f.write('not valid json\n')
            f.write('{"text": "valid2", "target_text": "ok2"}\n')
            path = f.name
        
        try:
            results = validate_jsonl_file(path)
            assert not results["valid"]
            assert results["total_samples"] == 3
            assert results["invalid_samples"] == 1
            assert any("Invalid JSON" in e["error"] for e in results["errors"])
        finally:
            Path(path).unlink()
    
    def test_mixed_valid_invalid_samples(self):
        samples = [
            {"text": "valid", "target_text": "ok"},
            {"target_text": "missing text"},  # Invalid
            {"text": "valid2", "target_text": "ok2"},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            path = f.name
        
        try:
            results = validate_jsonl_file(path)
            assert not results["valid"]
            assert results["valid_samples"] == 2
            assert results["invalid_samples"] == 1
        finally:
            Path(path).unlink()


class TestSpanEncodingValidation:
    """Tests for span encoding validation with actual encoders."""
    
    def test_arithmetic_span_encoding(self):
        """Test that arithmetic spans can be validated against encoders."""
        from fluxem.integration.pipeline import DomainEncoderRegistry
        
        registry = DomainEncoderRegistry()
        
        # Valid arithmetic span
        span = {"type": "arithmetic", "start": 0, "end": 3, "value": 123.0}
        errors = validate_span(span, "123", registry)
        # Should not have encoding errors (may have other errors depending on implementation)
        encoding_errors = [e for e in errors if "encoding" in e.field.lower()]
        assert len(encoding_errors) == 0
    
    def test_valid_sample_with_encoding_check(self):
        """Test full sample validation with encoding checks."""
        from fluxem.integration.pipeline import DomainEncoderRegistry
        
        registry = DomainEncoderRegistry()
        
        sample = {
            "text": "Calculate 42",
            "target_text": "42",
            "spans": [{"type": "arithmetic", "start": 10, "end": 12, "value": 42.0}]
        }
        
        result = validate_sample(sample, registry)
        # Should be valid (value can be encoded)
        assert result.valid or not any("encoding" in e.field for e in result.errors)
