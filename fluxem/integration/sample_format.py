"""
Mixed-sample JSONL format definition and validation for FluxEM hybrid training.

Canonical format:
{
    "text": "...",
    "spans": [
        {"type": "arithmetic", "start": 0, "end": 5, "value": 123.0},
        {"type": "phys_quantity", "start": 10, "end": 20, "value": 373.15, "dims": {"Theta": 1}}
    ],
    "target_text": "..."
}

The `spans` field is optional (token-only baseline uses none).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .tokenizer import DomainType


# Valid span types (maps to encoder names)
VALID_SPAN_TYPES = {
    "arithmetic",       # Numeric values/expressions
    "phys_quantity",    # Physical quantities with dimensions
    "phys_unit",        # Physical units
    "chem_formula",     # Chemical formulas
    "chem_reaction",    # Chemical reactions
    "math_complex",     # Complex numbers
    "math_rational",    # Rational numbers
    "math_vector",      # Vectors
    "math_matrix",      # Matrices
    "math_polynomial",  # Polynomials
    "logic_prop",       # Propositional logic
    "bio_dna",          # DNA sequences
    "bio_rna",          # RNA sequences
    "bio_protein",      # Protein sequences
    "music_pitch",      # Musical pitches
    "music_chord",      # Musical chords
    "music_atonal",     # Atonal pitch-class sets
}


@dataclass
class ValidationError:
    """Represents a validation error."""
    field: str
    message: str
    value: Any = None


@dataclass
class ValidationResult:
    """Result of sample validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __bool__(self):
        return self.valid


@dataclass
class Span:
    """A domain span within text."""
    type: str
    start: int
    end: int
    value: Any = None
    dims: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict:
        d = {"type": self.type, "start": self.start, "end": self.end}
        if self.value is not None:
            d["value"] = self.value
        if self.dims is not None:
            d["dims"] = self.dims
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Span":
        return cls(
            type=d.get("type", ""),
            start=d.get("start", 0),
            end=d.get("end", 0),
            value=d.get("value"),
            dims=d.get("dims"),
        )


@dataclass
class Sample:
    """A training/evaluation sample."""
    text: str
    target_text: str
    spans: List[Span] = field(default_factory=list)
    target_value: Any = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        d = {
            "text": self.text,
            "target_text": self.target_text,
        }
        if self.spans:
            d["spans"] = [s.to_dict() for s in self.spans]
        if self.target_value is not None:
            d["target_value"] = self.target_value
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Sample":
        spans = [Span.from_dict(s) for s in d.get("spans", [])]
        return cls(
            text=d.get("text", ""),
            target_text=d.get("target_text", ""),
            spans=spans,
            target_value=d.get("target_value"),
            metadata=d.get("metadata"),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, s: str) -> "Sample":
        return cls.from_dict(json.loads(s))


def validate_span(span: Dict, text: str, registry=None) -> List[ValidationError]:
    """
    Validate a single span.
    
    Args:
        span: Span dictionary
        text: Full text for bounds checking
        registry: Optional DomainEncoderRegistry for encoding validation
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields
    if "type" not in span:
        errors.append(ValidationError("span.type", "Missing required field 'type'"))
    elif span["type"] not in VALID_SPAN_TYPES:
        errors.append(ValidationError(
            "span.type",
            f"Unknown span type: {span['type']}. Valid types: {VALID_SPAN_TYPES}",
            span["type"]
        ))
    
    if "start" not in span:
        errors.append(ValidationError("span.start", "Missing required field 'start'"))
    elif not isinstance(span["start"], int) or span["start"] < 0:
        errors.append(ValidationError("span.start", "start must be a non-negative integer", span["start"]))
    
    if "end" not in span:
        errors.append(ValidationError("span.end", "Missing required field 'end'"))
    elif not isinstance(span["end"], int) or span["end"] < 0:
        errors.append(ValidationError("span.end", "end must be a non-negative integer", span["end"]))
    
    # Bounds checking
    if "start" in span and "end" in span:
        start, end = span["start"], span["end"]
        if isinstance(start, int) and isinstance(end, int):
            if start >= end:
                errors.append(ValidationError(
                    "span",
                    f"start ({start}) must be less than end ({end})"
                ))
            if end > len(text):
                errors.append(ValidationError(
                    "span",
                    f"end ({end}) exceeds text length ({len(text)})"
                ))
    
    # Try encoding if registry provided
    if registry is not None and "type" in span and span["type"] in VALID_SPAN_TYPES:
        try:
            # Map span type to DomainType
            type_mapping = {
                "arithmetic": DomainType.ARITHMETIC,
                "phys_quantity": DomainType.QUANTITY,
                "phys_unit": DomainType.UNIT,
                "chem_formula": DomainType.FORMULA,
                "chem_reaction": DomainType.REACTION,
                "math_complex": DomainType.COMPLEX,
                "math_rational": DomainType.RATIONAL,
                "math_vector": DomainType.VECTOR,
                "math_matrix": DomainType.MATRIX,
                "math_polynomial": DomainType.POLYNOMIAL,
                "logic_prop": DomainType.LOGICAL,
                "bio_dna": DomainType.DNA,
                "bio_rna": DomainType.RNA,
                "bio_protein": DomainType.PROTEIN,
                "music_pitch": DomainType.PITCH,
                "music_chord": DomainType.CHORD,
                "music_atonal": DomainType.ATONAL,
            }
            
            domain_type = type_mapping.get(span["type"])
            if domain_type:
                encoder = registry.get_encoder(domain_type)
                if encoder is not None:
                    # Try to encode the value
                    value = span.get("value")
                    if value is not None:
                        encoder.encode(value)
        except Exception as e:
            errors.append(ValidationError(
                "span.encoding",
                f"Failed to encode span: {e}",
                span
            ))
    
    return errors


def validate_sample(sample: Dict, registry=None) -> ValidationResult:
    """
    Validate a single sample.
    
    Args:
        sample: Sample dictionary
        registry: Optional DomainEncoderRegistry for encoding validation
    
    Returns:
        ValidationResult with errors/warnings
    """
    errors = []
    warnings = []
    
    # Required fields
    if "text" not in sample:
        errors.append(ValidationError("text", "Missing required field 'text'"))
    elif not isinstance(sample["text"], str):
        errors.append(ValidationError("text", "text must be a string", type(sample["text"])))
    
    if "target_text" not in sample:
        errors.append(ValidationError("target_text", "Missing required field 'target_text'"))
    elif not isinstance(sample["target_text"], str):
        errors.append(ValidationError("target_text", "target_text must be a string"))
    
    # Optional spans
    text = sample.get("text", "")
    if "spans" in sample:
        if not isinstance(sample["spans"], list):
            errors.append(ValidationError("spans", "spans must be a list"))
        else:
            for i, span in enumerate(sample["spans"]):
                span_errors = validate_span(span, text, registry)
                for err in span_errors:
                    err.field = f"spans[{i}].{err.field}"
                errors.extend(span_errors)
            
            # Check for overlapping spans
            spans = sample["spans"]
            for i, s1 in enumerate(spans):
                for j, s2 in enumerate(spans):
                    if i < j:
                        start1, end1 = s1.get("start", 0), s1.get("end", 0)
                        start2, end2 = s2.get("start", 0), s2.get("end", 0)
                        if start1 < end2 and start2 < end1:
                            warnings.append(f"Spans {i} and {j} overlap")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_jsonl_file(path: Union[str, Path], registry=None, max_errors: int = 100) -> Dict:
    """
    Validate a JSONL file.
    
    Args:
        path: Path to JSONL file
        registry: Optional DomainEncoderRegistry
        max_errors: Stop after this many errors
    
    Returns:
        Dict with validation summary
    """
    path = Path(path)
    
    results = {
        "file": str(path),
        "valid": True,
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "errors": [],
        "warnings": [],
    }
    
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if len(results["errors"]) >= max_errors:
                results["warnings"].append(f"Stopped after {max_errors} errors")
                break
            
            results["total_samples"] += 1
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                results["errors"].append({
                    "line": line_num,
                    "error": f"Invalid JSON: {e}"
                })
                results["invalid_samples"] += 1
                continue
            
            validation = validate_sample(sample, registry)
            
            if validation.valid:
                results["valid_samples"] += 1
            else:
                results["invalid_samples"] += 1
                for err in validation.errors:
                    results["errors"].append({
                        "line": line_num,
                        "field": err.field,
                        "error": err.message,
                        "value": err.value,
                    })
            
            for warn in validation.warnings:
                results["warnings"].append({"line": line_num, "warning": warn})
    
    results["valid"] = results["invalid_samples"] == 0
    
    return results


def main():
    """CLI for validating JSONL files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate FluxEM JSONL sample files")
    parser.add_argument("files", nargs="+", help="JSONL files to validate")
    parser.add_argument("--check-encoding", action="store_true", 
                       help="Validate that spans can be encoded")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show all errors")
    args = parser.parse_args()
    
    registry = None
    if args.check_encoding:
        from .pipeline import DomainEncoderRegistry
        registry = DomainEncoderRegistry()
    
    all_valid = True
    
    for file_path in args.files:
        print(f"\nValidating {file_path}...")
        results = validate_jsonl_file(file_path, registry)
        
        if results["valid"]:
            print(f"  ✓ Valid: {results['valid_samples']}/{results['total_samples']} samples")
        else:
            print(f"  ✗ Invalid: {results['invalid_samples']}/{results['total_samples']} samples have errors")
            all_valid = False
            
            if args.verbose:
                for err in results["errors"][:20]:
                    print(f"    Line {err['line']}: {err['field']} - {err['error']}")
        
        if results["warnings"]:
            print(f"  Warnings: {len(results['warnings'])}")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
