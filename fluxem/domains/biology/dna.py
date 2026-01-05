"""
DNA Sequence Encoder for Molecular Biology.

Encodes DNA sequences with their properties:
- Nucleotide composition
- GC content
- Molecular weight
- Thermodynamic stability
- Gene structure features
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
    log_encode_value,
    log_decode_value,
)

# =============================================================================
# DNA Data Structures
# =============================================================================

COMPLEMENTARY_BASES = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "a": "t",
    "t": "a",
    "g": "c",
    "c": "g",
}

NUCLEOTIDE_MW = {
    "A": 313.21,
    "T": 304.19,
    "G": 329.21,
    "C": 289.18,
    "a": 313.21,
    "t": 304.19,
    "g": 329.21,
    "c": 289.18,
}

BASE_ORDER = {"A": 0, "T": 1, "G": 2, "C": 3}

# Embedding layout within domain-specific region (64 dims starting at offset 8):
# dims 0-3:   Nucleotide composition [A, T, G, C] (normalized counts)
# dim 4:      GC content (ratio of G+C / total)
# dim 5:      Sequence length (log encoded)
# dim 6:      Molecular weight (log encoded)
# dim 7:      Melting temperature estimate
# dims 8-11:  Codon usage bias (for coding sequences)
# dims 12-15: Structural features (hairpins, repeats, etc.)
# dims 16-31: Reserved for genomic annotations
# dims 32-63: Sequence embedding (hash-based or learned)

COMPOSITION_OFFSET = 0
GC_CONTENT_OFFSET = 4
LENGTH_OFFSET = 5
MW_OFFSET = 6
TM_OFFSET = 7
CODON_USAGE_OFFSET = 8
STRUCTURAL_OFFSET = 12
GENOMIC_OFFSET = 16
SEQUENCE_EMBED_OFFSET = 32


@dataclass
class DNASequence:
    """DNA sequence with its properties."""

    sequence: str
    length: int
    gc_content: float
    molecular_weight: float
    is_coding: bool = False
    gene_name: Optional[str] = None
    organism: Optional[str] = None


def complement(sequence: str) -> str:
    """Get complementary DNA sequence."""
    return "".join(COMPLEMENTARY_BASES.get(base, "N") for base in sequence)


def reverse_complement(sequence: str) -> str:
    """Get reverse complement of DNA sequence."""
    return complement(sequence)[::-1]


def gc_content(sequence: str) -> float:
    """Calculate GC content ratio (0-1)."""
    sequence = sequence.upper()
    gc = sum(1 for base in sequence if base in "GC")
    return gc / len(sequence) if sequence else 0.0


def molecular_weight(sequence: str) -> float:
    """Calculate molecular weight of DNA sequence."""
    return sum(NUCLEOTIDE_MW.get(base, 0) for base in sequence.upper())


def melting_temperature(sequence: str) -> float:
    """Estimate melting temperature using Wallace rule."""
    # Simplified: Tm = 2(A+T) + 4(G+C) for short sequences
    sequence = sequence.upper()
    at = sum(1 for base in sequence if base in "AT")
    gc = sum(1 for base in sequence if base in "GC")
    return 2 * at + 4 * gc


# =============================================================================
# DNA Encoder
# =============================================================================


class DNAEncoder:
    """
    Encoder for DNA sequences.

    Embedding captures:
    - Nucleotide composition
    - GC content
    - Length and molecular weight
    - Thermodynamic properties
    - Potential coding features

    This enables operations on genetic information.
    """

    domain_tag = DOMAIN_TAGS["bio_dna"]
    domain_name = "bio_dna"

    def encode(self, sequence: str) -> Any:
        """
        Encode a DNA sequence to 128-dim embedding.

        Args:
            sequence: DNA string (e.g., 'ATGCCGTAGC')

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        sequence = sequence.upper().replace(" ", "")

        # Validate sequence
        if not all(base in "ATGCN" for base in sequence):
            raise ValueError(f"Invalid DNA sequence: {sequence}")

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Nucleotide composition (normalized)
        total = len(sequence)
        if total > 0:
            composition = [
                sequence.count("A") / total,
                sequence.count("T") / total,
                sequence.count("G") / total,
                sequence.count("C") / total,
            ]
        else:
            composition = [0.0, 0.0, 0.0, 0.0]

        for i, comp in enumerate(composition):
            emb = backend.at_add(emb, 8 + COMPOSITION_OFFSET + i, comp)

        # GC content
        emb = backend.at_add(emb, 8 + GC_CONTENT_OFFSET, gc_content(sequence))

        # Sequence length (log encoded)
        _, log_length = log_encode_value(total)
        emb = backend.at_add(emb, 8 + LENGTH_OFFSET, log_length)

        # Molecular weight (log encoded)
        mw = molecular_weight(sequence)
        _, log_mw = log_encode_value(mw)
        emb = backend.at_add(emb, 8 + MW_OFFSET, log_mw)

        # Melting temperature
        tm = melting_temperature(sequence) / 100.0  # Normalize
        emb = backend.at_add(emb, 8 + TM_OFFSET, tm)

        # Sequence embedding (hash-based for now)
        seq_hash = self._sequence_hash(sequence)
        for i, val in enumerate(seq_hash):
            emb = backend.at_add(emb, 8 + SEQUENCE_EMBED_OFFSET + i, val)

        return emb

    def decode(self, emb: Any) -> DNASequence:
        """
        Decode embedding back to DNA sequence (reconstruction).

        For sequences up to 32 bases, reconstruction uses stored base values.
        For longer sequences, only the first 32 bases are stored.
        """
        backend = get_backend()
        # Extract properties
        log_length = emb[8 + LENGTH_OFFSET].item()
        length = int(round(backend.exp(backend.array(log_length)).item()))

        gc = emb[8 + GC_CONTENT_OFFSET].item()

        # Reconstruct sequence from stored base values
        seq = self._reconstruct_sequence(length, emb)

        return DNASequence(
            sequence=seq,
            length=length,
            gc_content=gc,
            molecular_weight=backend.exp(backend.array(emb[8 + MW_OFFSET].item())).item(),
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid DNA sequence."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Biological Operations
    # ========================================================================

    def reverse_complement_embed(self, emb: Any) -> Any:
        """
        Compute embedding of reverse complement.
        """
        backend = get_backend()
        result = emb

        # Swap A and T in composition
        a_comp = result[8 + COMPOSITION_OFFSET + 0]
        t_comp = result[8 + COMPOSITION_OFFSET + 1]
        result = backend.at_add(result, 8 + COMPOSITION_OFFSET + 0, t_comp - a_comp)
        result = backend.at_add(result, 8 + COMPOSITION_OFFSET + 1, a_comp - t_comp)

        # Swap G and C
        g_comp = result[8 + COMPOSITION_OFFSET + 2]
        c_comp = result[8 + COMPOSITION_OFFSET + 3]
        result = backend.at_add(result, 8 + COMPOSITION_OFFSET + 2, c_comp - g_comp)
        result = backend.at_add(result, 8 + COMPOSITION_OFFSET + 3, g_comp - c_comp)

        return result

    def similarity(self, emb1: Any, emb2: Any) -> float:
        """
        Compute similarity between two DNA sequences.

        Based on composition and GC content.
        """
        backend = get_backend()
        # Compare composition
        comp1 = emb1[8 + COMPOSITION_OFFSET : 8 + COMPOSITION_OFFSET + 4]
        comp2 = emb2[8 + COMPOSITION_OFFSET : 8 + COMPOSITION_OFFSET + 4]
        comp_sim = 1.0 - backend.abs(comp1 - comp2).mean().item() / 2.0

        # Compare GC content
        gc1 = emb1[8 + GC_CONTENT_OFFSET].item()
        gc2 = emb2[8 + GC_CONTENT_OFFSET].item()
        gc_sim = 1.0 - abs(gc1 - gc2)

        return (comp_sim + gc_sim) / 2.0

    def hamming_distance_estimate(self, emb1: Any, emb2: Any) -> float:
        """
        Estimate Hamming distance between sequences.

        Based on composition difference.
        """
        backend = get_backend()
        log_len1 = emb1[8 + LENGTH_OFFSET].item()
        log_len2 = emb2[8 + LENGTH_OFFSET].item()

        len1 = backend.exp(backend.array(log_len1)).item()
        len2 = backend.exp(backend.array(log_len2)).item()

        # Composition difference
        comp_diff = (
            backend.abs(
                emb1[8 + COMPOSITION_OFFSET : 8 + COMPOSITION_OFFSET + 4]
                - emb2[8 + COMPOSITION_OFFSET : 8 + COMPOSITION_OFFSET + 4]
            )
            .sum()
            .item()
        )

        # Estimate mutations
        avg_len = (len1 + len2) / 2.0
        estimated_mutations = comp_diff * avg_len / 2.0

        return estimated_mutations

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _sequence_hash(self, sequence: str) -> List[float]:
        """
        Encode the sequence as a list of base values for reconstruction.

        Each base is encoded as a distinct value:
        - A = 0.25
        - T = 0.50
        - G = 0.75
        - C = 1.00
        - Empty/padding = 0.0

        We have 32 dimensions, so sequences up to 32 bases can be stored directly.
        For longer sequences, we store the first 32 bases (truncated).
        """
        BASE_VALUES = {"A": 0.25, "T": 0.50, "G": 0.75, "C": 1.00}
        values = []
        for i in range(32):
            if i < len(sequence):
                values.append(BASE_VALUES.get(sequence[i], 0.0))
            else:
                values.append(0.0)  # Padding for shorter sequences
        return values

    def _reconstruct_sequence(self, length: int, emb: Any) -> str:
        """
        Reconstruct a sequence from the embedding.

        Decodes base values stored in the sequence embedding region:
        - 0.25 -> A
        - 0.50 -> T
        - 0.75 -> G
        - 1.00 -> C
        """
        if length <= 0:
            return ""

        # Decode bases from the sequence embedding region
        VALUE_TO_BASE = {0.25: "A", 0.50: "T", 0.75: "G", 1.00: "C"}
        bases = []

        # Only reconstruct up to the stored length (max 32 bases)
        reconstruct_length = min(length, 32)

        for i in range(reconstruct_length):
            val = emb[8 + SEQUENCE_EMBED_OFFSET + i].item()
            # Find closest matching base value
            closest_base = "A"  # Default
            closest_dist = float("inf")
            for base_val, base in VALUE_TO_BASE.items():
                dist = abs(val - base_val)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_base = base
            bases.append(closest_base)

        return "".join(bases)


def translate_dna_to_protein(dna_seq: str) -> str:
    """
    Translate DNA sequence to protein (standard genetic code).

    Args:
        dna_seq: DNA sequence (must be multiple of 3)

    Returns:
        Amino acid sequence
    """
    # Standard genetic code table
    CODON_TABLE = {
        "TTT": "F",
        "TTC": "F",
        "TTA": "L",
        "TTG": "L",
        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "TAT": "Y",
        "TAC": "Y",
        "TAA": "*",
        "TAG": "*",
        "TGT": "C",
        "TGC": "C",
        "TGA": "*",
        "TGG": "W",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "CAT": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "ATT": "I",
        "ATC": "I",
        "ATA": "I",
        "ATG": "M",
        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "AAT": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "AGT": "S",
        "AGC": "S",
        "AGA": "R",
        "AGG": "R",
        "GTT": "V",
        "GTC": "V",
        "GTA": "V",
        "GTG": "V",
        "GCT": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",
        "GAT": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }

    dna_seq = dna_seq.upper()
    protein = []

    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i : i + 3]
        if len(codon) == 3:
            aa = CODON_TABLE.get(codon, "X")  # X for unknown
            protein.append(aa)

    return "".join(protein)
