"""Biology domain - DNA sequences, GC content, translation.

This module provides deterministic biology computations.
"""

from typing import Any, Dict, List, Optional, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

# DNA base complements
COMPLEMENT_MAP = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

# Nucleotide molecular weights (g/mol)
NUCLEOTIDE_MW = {
    'A': 313.21,  # dAMP
    'T': 304.19,  # dTMP
    'G': 329.21,  # dGMP
    'C': 289.18,  # dCMP
}

# Standard genetic code (DNA codons to amino acids)
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


# =============================================================================
# Core Functions
# =============================================================================

def gc_content(sequence: str) -> float:
    """Calculate GC content ratio (0-1) for a DNA sequence.

    Args:
        sequence: DNA sequence string (A, T, G, C)

    Returns:
        GC content as float between 0 and 1
    """
    sequence = sequence.upper().strip()
    if not sequence:
        return 0.0
    gc_count = sum(1 for base in sequence if base in ('G', 'C'))
    return gc_count / len(sequence)


def molecular_weight(sequence: str) -> float:
    """Calculate molecular weight of a DNA sequence.

    Args:
        sequence: DNA sequence string

    Returns:
        Molecular weight in g/mol
    """
    sequence = sequence.upper().strip()
    weight = sum(NUCLEOTIDE_MW.get(base, 0) for base in sequence)
    return round(weight, 2)


def complement(sequence: str) -> str:
    """Generate complementary DNA sequence.

    Args:
        sequence: DNA sequence string

    Returns:
        Complementary sequence (5' to 3')
    """
    sequence = sequence.upper().strip()
    return ''.join(COMPLEMENT_MAP.get(base, base) for base in sequence)


def reverse_complement(sequence: str) -> str:
    """Generate reverse complement of a DNA sequence.

    Args:
        sequence: DNA sequence string

    Returns:
        Reverse complement sequence
    """
    return complement(sequence)[::-1]


def reverse_complement_gc(sequence: str) -> float:
    """Compute GC content of the reverse complement.

    Args:
        sequence: DNA sequence string

    Returns:
        GC content of reverse complement
    """
    # GC content is the same for sequence and its reverse complement
    return gc_content(sequence)


def translate(sequence: str) -> str:
    """Translate a DNA sequence into amino acid sequence.

    Uses the standard genetic code. Stop codons are marked with '*'.

    Args:
        sequence: DNA sequence string (length should be multiple of 3)

    Returns:
        Amino acid sequence (single-letter codes)
    """
    sequence = sequence.upper().strip()
    protein = []
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        aa = CODON_TABLE.get(codon, 'X')  # X for unknown
        if aa == '*':  # Stop codon
            break
        protein.append(aa)
    return ''.join(protein)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_sequence(args):
    if isinstance(args, dict):
        return args.get("sequence", args.get("seq", args.get("dna", list(args.values())[0])))
    return str(args)


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register biology tools in the registry."""

    registry.register(ToolSpec(
        name="biology_gc_content",
        function=lambda args: gc_content(_parse_sequence(args)),
        description="Calculates GC content ratio (0-1) for a DNA sequence.",
        parameters={
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA sequence (A, T, G, C)"}
            },
            "required": ["sequence"]
        },
        returns="GC content as float between 0 and 1",
        examples=[
            {"input": {"sequence": "GATTACA"}, "output": 0.2857},
            {"input": {"sequence": "ATGC"}, "output": 0.5},
        ],
        domain="biology",
        tags=["dna", "gc", "content", "sequence"],
    ))

    registry.register(ToolSpec(
        name="biology_mw",
        function=lambda args: molecular_weight(_parse_sequence(args)),
        description="Calculates molecular weight of a DNA sequence.",
        parameters={
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA sequence"}
            },
            "required": ["sequence"]
        },
        returns="Molecular weight in g/mol",
        examples=[
            {"input": {"sequence": "ATCG"}, "output": 1235.79},
        ],
        domain="biology",
        tags=["dna", "molecular", "weight"],
    ))

    registry.register(ToolSpec(
        name="biology_complement",
        function=lambda args: complement(_parse_sequence(args)),
        description="Generates complementary DNA sequence (A↔T, G↔C).",
        parameters={
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA sequence"}
            },
            "required": ["sequence"]
        },
        returns="Complementary DNA sequence",
        examples=[
            {"input": {"sequence": "ATCG"}, "output": "TAGC"},
            {"input": {"sequence": "GATTACA"}, "output": "CTAATGT"},
        ],
        domain="biology",
        tags=["dna", "complement", "base pairing"],
    ))

    registry.register(ToolSpec(
        name="biology_reverse_complement_gc",
        function=lambda args: reverse_complement_gc(_parse_sequence(args)),
        description="Computes GC content of the reverse complement of a DNA sequence.",
        parameters={
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA sequence"}
            },
            "required": ["sequence"]
        },
        returns="GC content as float between 0 and 1",
        examples=[
            {"input": {"sequence": "GATTACA"}, "output": 0.2857},
        ],
        domain="biology",
        tags=["dna", "reverse complement", "gc"],
    ))

    registry.register(ToolSpec(
        name="biology_translate",
        function=lambda args: translate(_parse_sequence(args)),
        description="Translates a DNA sequence into amino acid sequence using the standard genetic code.",
        parameters={
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA sequence (length should be multiple of 3)"}
            },
            "required": ["sequence"]
        },
        returns="Amino acid sequence (single-letter codes)",
        examples=[
            {"input": {"sequence": "ATGGCCATT"}, "output": "MAI"},
            {"input": {"sequence": "ATGTAA"}, "output": "M"},
        ],
        domain="biology",
        tags=["dna", "protein", "translation", "codon"],
    ))
