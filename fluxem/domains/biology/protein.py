"""
Protein Sequence Encoder for Molecular Biology.

Encodes protein sequences with their properties:
- Amino acid composition
- Molecular weight
- Isoelectric point
- Hydrophobicity
- Secondary structure propensity
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
# Amino Acid Properties
# =============================================================================

AMINO_ACIDS = {
    "A": ("Alanine", 89.09, 6.01, 1.8),
    "R": ("Arginine", 174.20, 10.76, -4.5),
    "N": ("Asparagine", 132.12, 5.41, -3.5),
    "D": ("Aspartic acid", 133.10, 2.77, -3.5),
    "C": ("Cysteine", 121.16, 5.07, 2.5),
    "E": ("Glutamic acid", 147.13, 3.22, -3.5),
    "Q": ("Glutamine", 146.15, 5.65, -3.5),
    "G": ("Glycine", 75.07, 5.97, -0.4),
    "H": ("Histidine", 155.16, 7.59, -3.2),
    "I": ("Isoleucine", 131.18, 6.02, 4.5),
    "L": ("Leucine", 131.18, 5.98, 3.8),
    "K": ("Lysine", 146.19, 9.74, -3.9),
    "M": ("Methionine", 149.21, 5.74, 1.9),
    "F": ("Phenylalanine", 165.19, 5.48, 2.8),
    "P": ("Proline", 115.13, 6.30, -1.6),
    "S": ("Serine", 105.09, 5.68, -0.8),
    "T": ("Threonine", 119.12, 5.60, -0.7),
    "W": ("Tryptophan", 204.23, 5.89, -0.9),
    "Y": ("Tyrosine", 181.19, 5.66, -1.3),
    "V": ("Valine", 117.15, 5.96, 4.2),
}

# Embedding layout within domain-specific region (64 dims starting at offset 8):
# dims 0-19:  Amino acid composition (20 amino acids, normalized)
# dim 20:     Sequence length (log encoded)
# dim 21:     Molecular weight (log encoded)
# dim 22:     Estimated isoelectric point
# dim 23:     Average hydrophobicity
# dim 24:     Instability index
# dim 25:     Aliphatic index
# dims 26-27: Secondary structure propensity [helix, sheet]
# dims 28-31: Functional motifs
# dims 32-63: Sequence embedding (hash-based or learned)

COMPOSITION_OFFSET = 0
LENGTH_OFFSET = 20
MW_OFFSET = 21
PI_OFFSET = 22
HYDROPHOBICITY_OFFSET = 23
INSTABILITY_OFFSET = 24
ALIPHATIC_OFFSET = 25
SECONDARY_OFFSET = 26
MOTIF_OFFSET = 28
SEQUENCE_EMBED_OFFSET = 32


@dataclass
class ProteinSequence:
    """Protein sequence with its properties."""

    sequence: str
    length: int
    molecular_weight: float
    isoelectric_point: float
    hydrophobicity: float
    protein_name: Optional[str] = None
    organism: Optional[str] = None


# =============================================================================
# Protein Encoder
# =============================================================================


class ProteinEncoder:
    """
    Encoder for protein sequences.

    Embedding captures:
    - Amino acid composition
    - Physicochemical properties
    - Structural propensity
    - Functional motifs
    """

    domain_tag = DOMAIN_TAGS["bio_protein"]
    domain_name = "bio_protein"

    def encode(self, sequence: str) -> Any:
        """
        Encode a protein sequence to 128-dim embedding.

        Args:
            sequence: Protein string in one-letter code (e.g., 'MVLSPADKTN')

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        sequence = sequence.upper()

        # Validate sequence
        if not all(aa in AMINO_ACIDS or aa == "X" for aa in sequence):
            raise ValueError(f"Invalid protein sequence: {sequence}")

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Amino acid composition (normalized)
        total = len(sequence)
        aa_list = list(AMINO_ACIDS.keys())

        if total > 0:
            for i, aa in enumerate(aa_list):
                count = sequence.count(aa)
                emb = backend.at_add(emb, 8 + COMPOSITION_OFFSET + i, count / total)

        # Sequence length (log encoded)
        _, log_length = log_encode_value(total)
        emb = backend.at_add(emb, 8 + LENGTH_OFFSET, log_length)

        # Molecular weight (log encoded)
        mw = self._molecular_weight(sequence)
        _, log_mw = log_encode_value(mw)
        emb = backend.at_add(emb, 8 + MW_OFFSET, log_mw)

        # Isoelectric point estimate
        pi = self._isoelectric_point(sequence)
        emb = backend.at_add(emb, 8 + PI_OFFSET, pi / 14.0)  # Normalize [0,1]

        # Average hydrophobicity
        hydro = self._average_hydrophobicity(sequence)
        emb = backend.at_add(emb, 8 + HYDROPHOBICITY_OFFSET, hydro / 5.0)  # Normalize

        # Instability index
        instability = self._instability_index(sequence)
        emb = backend.at_add(emb, 8 + INSTABILITY_OFFSET, instability / 100.0)

        # Aliphatic index
        aliphatic = self._aliphatic_index(sequence)
        emb = backend.at_add(emb, 8 + ALIPHATIC_OFFSET, aliphatic / 150.0)

        # Secondary structure propensity
        helix_prop, sheet_prop = self._secondary_structure_propensity(sequence)
        emb = backend.at_add(emb, 8 + SECONDARY_OFFSET, helix_prop)
        emb = backend.at_add(emb, 8 + SECONDARY_OFFSET + 1, sheet_prop)

        # Sequence embedding (hash-based)
        seq_hash = self._sequence_hash(sequence)
        for i, val in enumerate(seq_hash):
            emb = backend.at_add(emb, 8 + SEQUENCE_EMBED_OFFSET + i, val)

        return emb

    def decode(self, emb: Any) -> ProteinSequence:
        """
        Decode embedding back to protein sequence.

        Note: Approximate reconstruction based on composition.
        """
        backend = get_backend()
        # Extract properties
        log_length = emb[8 + LENGTH_OFFSET].item()
        length = int(round(backend.exp(backend.array(log_length)).item()))

        # Get composition
        aa_list = list(AMINO_ACIDS.keys())
        composition = {}
        for i, aa in enumerate(aa_list):
            count = emb[8 + COMPOSITION_OFFSET + i].item()
            if count > 0:
                composition[aa] = int(round(count * length))

        # Reconstruct sequence with matching composition
        seq = self._reconstruct_sequence(composition, length)

        return ProteinSequence(
            sequence=seq,
            length=length,
            molecular_weight=backend.exp(backend.array(emb[8 + MW_OFFSET].item())).item(),
            isoelectric_point=emb[8 + PI_OFFSET].item() * 14.0,
            hydrophobicity=emb[8 + HYDROPHOBICITY_OFFSET].item() * 5.0,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid protein sequence."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Biological Operations
    # ========================================================================

    def similarity(self, emb1: Any, emb2: Any) -> float:
        """
        Compute similarity between two protein sequences.

        Based on composition and physicochemical properties.
        """
        backend = get_backend()
        # Compare composition
        comp_diff = (
            backend.abs(
                emb1[8 + COMPOSITION_OFFSET : 8 + COMPOSITION_OFFSET + 20]
                - emb2[8 + COMPOSITION_OFFSET : 8 + COMPOSITION_OFFSET + 20]
            )
            .mean()
            .item()
        )

        comp_sim = 1.0 - comp_diff

        # Compare hydrophobicity
        hydro1 = emb1[8 + HYDROPHOBICITY_OFFSET].item()
        hydro2 = emb2[8 + HYDROPHOBICITY_OFFSET].item()
        hydro_sim = 1.0 - abs(hydro1 - hydro2) / 5.0

        return (comp_sim + hydro_sim) / 2.0

    def alignment_score(self, emb1: Any, emb2: Any) -> float:
        """
        Estimate sequence alignment score.

        Simplified BLOSUM-like scoring based on composition.
        """
        return self.similarity(emb1, emb2)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight of protein."""
        total = 0.0
        for aa in sequence:
            if aa in AMINO_ACIDS:
                total += AMINO_ACIDS[aa][1]  # MW
        return total

    def _isoelectric_point(self, sequence: str) -> float:
        """
        Estimate isoelectric point (simplified).

        Uses Henderson-Hasselbalch approximation.
        """
        # Count charged residues
        pos_charges = sequence.count("K") + sequence.count("R") + sequence.count("H")
        neg_charges = sequence.count("D") + sequence.count("E")

        # Approximate pI based on charge balance
        if pos_charges == 0 and neg_charges == 0:
            return 7.0  # Neutral

        # Simplified: pI ~ 7.0 when balanced, shifts based on excess charge
        charge_diff = pos_charges - neg_charges
        pi = 7.0 + charge_diff * 0.5

        return max(2.0, min(12.0, pi))

    def _average_hydrophobicity(self, sequence: str) -> float:
        """Calculate average hydrophobicity using Kyte-Doolittle scale."""
        hydro_values = [AMINO_ACIDS[aa][3] for aa in sequence if aa in AMINO_ACIDS]
        return sum(hydro_values) / len(hydro_values) if hydro_values else 0.0

    def _instability_index(self, sequence: str) -> float:
        """
        Calculate instability index (Guruprasad et al., 1990).

        Higher values indicate less stable proteins.
        """
        # Simplified implementation
        # Count destabilizing residues (S, P, G, N)
        destabilizing = (
            sequence.count("S")
            + sequence.count("P")
            + sequence.count("G")
            + sequence.count("N")
        )
        return (destabilizing / len(sequence) * 100) if sequence else 50.0

    def _aliphatic_index(self, sequence: str) -> float:
        """
        Calculate aliphatic index (Ikai, 1980).

        Measures thermal stability based on aliphatic side chains.
        """
        a = sequence.count("A")
        v = sequence.count("V")
        i = sequence.count("I")
        l = sequence.count("L")
        total = len(sequence)

        if total == 0:
            return 0.0

        ai = (a + 2.9 * v + 3.9 * (i + l)) / total * 100
        return ai

    def _secondary_structure_propensity(self, sequence: str) -> Tuple[float, float]:
        """
        Estimate secondary structure propensity.

        Returns (helix_propensity, sheet_propensity).
        """
        # Chou-Fasman propensities (simplified)
        helix_favoring = "ALMEKQR"
        sheet_favoring = "VICYFWT"

        total = len(sequence)
        if total == 0:
            return (0.0, 0.0)

        helix_count = sum(1 for aa in sequence if aa in helix_favoring)
        sheet_count = sum(1 for aa in sequence if aa in sheet_favoring)

        return (helix_count / total, sheet_count / total)

    def _sequence_hash(self, sequence: str) -> List[float]:
        """Create a deterministic hash embedding of the sequence."""
        values = []
        for i in range(32):
            val = 0.0
            for j, aa in enumerate(sequence):
                aa_code = ord(aa)
                val += aa_code * ((i + 1) ** j)
            values.append((val % 10000) / 10000.0)
        return values

    def _reconstruct_sequence(
        self, composition: Dict[str, int], total_length: int
    ) -> str:
        """Reconstruct sequence from composition."""
        seq_parts = []
        for aa, count in composition.items():
            seq_parts.extend([aa] * count)

        # Adjust to match exact length
        while len(seq_parts) < total_length:
            seq_parts.append("A")  # Fill with alanine

        seq = "".join(seq_parts[:total_length])
        return seq
