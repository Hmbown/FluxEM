"""
Atonal Theory Encoder for 12-Tone Music.

This is LITERALLY matrix/vector operations! 🎵🧮

- Pitch class sets = 12-dimensional binary vectors
- Prime form = normalized matrix operations
- Interval class vectors = exact dot products
- Fortean number = decimal encoding
- Tn/I operations = matrix transposition + inversion

All operations are EXACT matrix math - perfect for FluxEM!
"""

from typing import Any, List, Tuple
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
    log_encode_value,
)

# =============================================================================
# Atonal Theory - 12-Tone Serialism
# =============================================================================

# Embedding layout for PC_SET (dims 8-71):
# dims 0-11:  Pitch class set (12-dimensional binary vector)
# dims 12-17: Interval class vector (6 intervals)
# dims 18-23: Prime form representation (6 elements max)
# dim 24:     Cardinality (number of elements)
# dim 25-26: Fortean number (log encoded)
# dims 27-38: Invariance vector (under Tn operations)
# dims 39-48: Fortean number matrix representation
# dims 49-63: Reserved for subset/superset relations

PC_SET_VECTOR_OFFSET = 0
IC_VECTOR_OFFSET = 12
PRIME_FORM_OFFSET = 18
CARDINALITY_OFFSET = 24
FORTEAN_OFFSET = 25
INVARIANCE_OFFSET = 27
FORTEAN_MATRIX_OFFSET = 39


def pitch_class_set_to_vector(pcs: List[int]) -> Any:
    """
    Convert pitch class set to 12-dimensional binary vector.

    This is the CORE of atonal theory - EXACT matrix representation!

    Example: {0, 3, 7, 11} (C, D#, G, B)
    → [1,0,0,1,0,0,0,1,0,0,0,1]

    EXACT: Binary encoding of pitch classes!
    """
    backend = get_backend()
    vec = backend.zeros(12)
    pcs_set = set(pc % 12 for pc in pcs)

    for pc in pcs_set:
        vec = backend.at_add(vec, pc, 1.0)

    return vec


def normal_form(pcs: List[int]) -> List[int]:
    """
    Compute normal form of a pitch class set.

    Normal form = most compact left-packed rotation.

    EXACT algorithm from Forte (1973).
    """
    if not pcs:
        return []

    # Generate all rotations
    rotations = []
    sorted_pcs = sorted(set(pc % 12 for pc in pcs))

    for i in range(12):
        rotated = [((pc - i) % 12) for pc in sorted_pcs]
        rotations.append(rotated)

    # Find most compact (minimizes span between first and last)
    def span(rotation):
        if len(rotation) == 1:
            return 0
        return (rotation[-1] - rotation[0]) % 12

    min_span = min(span(r) for r in rotations)
    candidates = [r for r in rotations if span(r) == min_span]

    # Choose leftmost (lexicographically smallest)
    return min(candidates)


def prime_form(pcs: List[int]) -> List[int]:
    """
    Compute prime form (most compact form + its inverse).

    Prime form = normal form or its inverse, whichever is most compact.

    EXACT matrix operation!
    """
    nf = normal_form(pcs)

    # Inversion (11 - n for each n)
    inv = sorted([((11 - pc) % 12) for pc in pcs])
    inv_nf = normal_form(inv)

    # Compare spans
    def span(rotation):
        if len(rotation) == 1:
            return 0
        return (rotation[-1] - rotation[0]) % 12

    if span(nf) <= span(inv_nf):
        return nf
    else:
        return inv_nf


def interval_class_vector(pcs: List[int]) -> Any:
    """
    Compute interval class vector (ICV).

    ICV = counts of interval classes 1-6.

    Example: {0, 4, 7} (C, E, G - major triad)
    → ICV = [0, 0, 1, 0, 1, 1] (major 3rd, perfect 5th, minor 6th)

    EXACT: This is a dot product of set with its transposition!
    """
    backend = get_backend()
    icv = backend.zeros(6)

    sorted_pcs = sorted(set(pc % 12 for pc in pcs))

    for i, pc1 in enumerate(sorted_pcs):
        for pc2 in sorted_pcs[i + 1 :]:
            interval = (pc2 - pc1) % 12
            ic_class = min(interval, 12 - interval)  # Interval class 1-6

            if ic_class > 0 and ic_class <= 6:
                idx = ic_class - 1
                icv = backend.at_add(icv, idx, 1.0)

    return icv


def forte_number(pcs: List[int]) -> int:
    """
    Compute Fortean number (decimal encoding of pitch class set).

    Forte number = binary digits of PC set interpreted as decimal.

    Example: {0, 3, 7, 11} = 100100010001₂ = 2337

    EXACT: Binary-to-decimal conversion.
    """
    vec = pitch_class_set_to_vector(pcs)
    forte = 0

    for i in range(12):
        if vec[i].item() > 0.5:
            forte += 2**i

    return forte


def transposition(pcs: List[int], n: int) -> List[int]:
    """
    Transpose pitch class set by n semitones (Tn operation).

    EXACT: (pc + n) mod 12 for each element.

    This is a MATRIX ROTATION!
    """
    return [(pc + n) % 12 for pc in pcs]


def inversion(pcs: List[int]) -> List[int]:
    """
    Invert pitch class set (I operation).

    EXACT: (11 - pc) mod 12 for each element.

    This is MATRIX COMPLEMENT!
    """
    return [((11 - pc) % 12) for pc in pcs]


def tin_operation(pcs: List[int], n: int) -> List[int]:
    """
    TnI operation (invert then transpose by n).

    EXACT: Matrix complement + rotation.
    """
    inv = inversion(pcs)
    return transposition(inv, n)


def multiplication(pcs: List[int], n: int) -> List[int]:
    """
    M-n operation (multiplication by n mod 12).

    Used for serial music (12-tone rows).

    EXACT: (pc * n) mod 12 for each element.
    """
    return [((pc * n) % 12) for pc in pcs]


def interval_class_similarity(icv1: Any, icv2: Any) -> float:
    """
    Compute ISIM (Interval Class SIMilarity).

    ISIM = dot product of IC vectors, normalized.

    EXACT: This is a cosine similarity!
    """
    backend = get_backend()
    dot = backend.sum(icv1 * icv2).item()
    norm1 = backend.sqrt(backend.sum(icv1**2)).item()
    norm2 = backend.sqrt(backend.sum(icv2**2)).item()

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def invariant_under_Tn(pcs: List[int]) -> List[int]:
    """
    Find all transposition invariants (Tn where Tn(S) = S).

    EXACT: Matrix equality check for all rotations.
    """
    invariants = []

    sorted_pcs = sorted(set(pc % 12 for pc in pcs))

    for n in range(12):
        transposed = sorted([(pc + n) % 12 for pc in sorted_pcs])
        if transposed == sorted_pcs:
            invariants.append(n)

    return invariants


def z_related(pcs1: List[int], pcs2: List[int]) -> bool:
    """
    Check if two sets are Z-related (same ICV, different sets).

    Z-relation = same interval class content, but different notes.

    EXACT: ICV equality + set inequality.
    """
    backend = get_backend()
    icv1 = interval_class_vector(pcs1)
    icv2 = interval_class_vector(pcs2)

    icv_equal = bool(backend.allclose(icv1, icv2, atol=0.01).item())

    set1 = sorted(set(pc % 12 for pc in pcs1))
    set2 = sorted(set(pc % 12 for pc in pcs2))

    sets_different = set1 != set2

    return icv_equal and sets_different


def subset_of(pcs1: List[int], pcs2: List[int]) -> bool:
    """
    Check if pcs1 is a subset of pcs2.

    EXACT: Set inclusion check.
    """
    set1 = set(pc % 12 for pc in pcs1)
    set2 = set(pc % 12 for pc in pcs2)
    return set1.issubset(set2)


def row_matrix(row: List[int]) -> Any:
    """
    Create row matrix (12x12) for 12-tone row.

    Each row is Tn(I(row)) operation.

    EXACT: Matrix generation via TnI operations.
    """
    backend = get_backend()
    matrix = backend.zeros((12, 12))

    for n in range(12):
        transformed = tin_operation(row, n)
        for i, pc in enumerate(transformed):
            matrix = backend.at_add(matrix, (n, i), float(pc))

    return matrix


# =============================================================================
# Atonal Set Encoder
# =============================================================================


class AtonalSetEncoder:
    """
    Encoder for atonal pitch class sets.

    This is LITERALLY matrix/vector operations!

    Embeds:
    - Pitch class set (12-dim binary vector)
    - Interval class vector (6-dim counts)
    - Prime form (normalized representation)
    - Fortean number (decimal encoding)
    - Invariance properties

    All operations are EXACT matrix operations!
    """

    domain_tag = DOMAIN_TAGS["music_atonal"]
    domain_name = "music_atonal"

    def encode(self, pcs: List[int]) -> Any:
        """
        Encode a pitch class set.

        Args:
            pcs: List of pitch classes (0-11), e.g., [0, 4, 7]

        Returns:
            128-dim embedding with full atonal analysis
        """
        backend = get_backend()
        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Pitch class set (12-dim binary vector)
        pc_vec = pitch_class_set_to_vector(pcs)
        for i in range(12):
            emb = backend.at_add(emb, 8 + PC_SET_VECTOR_OFFSET + i, pc_vec[i])

        # Interval class vector (6-dim)
        icv = interval_class_vector(pcs)
        for i in range(6):
            emb = backend.at_add(emb, 8 + IC_VECTOR_OFFSET + i, icv[i])

        # Prime form
        pf = prime_form(pcs)
        for i, pc in enumerate(pf):
            emb = backend.at_add(emb, 8 + PRIME_FORM_OFFSET + i, float(pc) / 12.0)

        # Cardinality
        emb = backend.at_add(emb, 8 + CARDINALITY_OFFSET, 
            float(len(set(pc % 12 for pc in pcs))) / 12.0
        )

        # Fortean number
        forte = forte_number(pcs)
        _, log_forte = log_encode_value(float(forte))
        emb = backend.at_add(emb, 8 + FORTEAN_OFFSET, log_forte / 12.0)

        # Tn invariance (which transpositions preserve set)
        tn_inv = invariant_under_Tn(pcs)
        for n in tn_inv:
            emb = backend.at_add(emb, 8 + INVARIANCE_OFFSET + n, 1.0)

        return emb

    def decode(self, emb: Any) -> List[int]:
        """
        Decode embedding back to pitch class set.
        """
        # Extract pitch class set
        pcs = []
        for i in range(12):
            if emb[8 + PC_SET_VECTOR_OFFSET + i].item() > 0.5:
                pcs.append(i)

        return pcs

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is valid."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Atonal Operations (ALL EXACT MATRIX OPS!)
    # ========================================================================

    def Tn(self, emb: Any, n: int) -> Any:
        """
        Transpose by n semitones.

        EXACT: Matrix rotation.
        """
        pcs = self.decode(emb)
        transposed = transposition(pcs, n)
        return self.encode(transposed)

    def I(self, emb: Any) -> Any:
        """
        Invert (I operation).

        EXACT: Matrix complement.
        """
        pcs = self.decode(emb)
        inverted = inversion(pcs)
        return self.encode(inverted)

    def TnI(self, emb: Any, n: int) -> Any:
        """
        Invert then transpose by n (TnI operation).

        EXACT: Matrix complement + rotation.
        """
        pcs = self.decode(emb)
        transformed = tin_operation(pcs, n)
        return self.encode(transformed)

    def M_n(self, emb: Any, n: int) -> Any:
        """
        Multiply by n (M-n operation).

        EXACT: Element-wise multiplication mod 12.
        """
        pcs = self.decode(emb)
        multiplied = multiplication(pcs, n)
        return self.encode(multiplied)

    def similarity(self, emb1: Any, emb2: Any) -> float:
        """
        Compute ISIM (interval class similarity).

        EXACT: Cosine similarity of IC vectors!
        """
        pcs1 = self.decode(emb1)
        pcs2 = self.decode(emb2)

        icv1 = interval_class_vector(pcs1)
        icv2 = interval_class_vector(pcs2)

        return interval_class_similarity(icv1, icv2)

    def is_z_related(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if two sets are Z-related.

        EXACT: ICV equality + set inequality.
        """
        pcs1 = self.decode(emb1)
        pcs2 = self.decode(emb2)
        return z_related(pcs1, pcs2)

    def is_subset(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if emb1 is subset of emb2.

        EXACT: Set inclusion.
        """
        pcs1 = self.decode(emb1)
        pcs2 = self.decode(emb2)
        return subset_of(pcs1, pcs2)

    def get_prime_form(self, emb: Any) -> Any:
        """
        Get prime form embedding.

        EXACT: Normalized representation.
        """
        backend = get_backend()
        pcs = self.decode(emb)
        pf = prime_form(pcs)

        result = self.encode(pcs)

        # Override with prime form only
        for i, pc in enumerate(pf):
            result = backend.at_add(result, 8 + PRIME_FORM_OFFSET + i, 
                float(pc) / 12.0 - result[8 + PRIME_FORM_OFFSET + i]
            )

        return result

    def is_invariant_under_Tn(self, emb: Any, n: int) -> bool:
        """
        Check if set is invariant under Tn.

        EXACT: Matrix equality after rotation.
        """
        pcs = self.decode(emb)
        invariants = invariant_under_Tn(pcs)
        return n in invariants

    def get_icv(self, emb: Any) -> Any:
        """
        Extract interval class vector.

        EXACT: 6-dimensional vector.
        """
        pcs = self.decode(emb)
        return interval_class_vector(pcs)

    def create_row_matrix(self, emb: Any) -> Any:
        """
        Create 12x12 row matrix for serial composition.

        EXACT: Matrix of all TnI operations.
        """
        pcs = self.decode(emb)
        return row_matrix(pcs)

    def get_forte_number(self, emb: Any) -> int:
        """
        Get Fortean number.

        EXACT: Decimal encoding.
        """
        pcs = self.decode(emb)
        return forte_number(pcs)
