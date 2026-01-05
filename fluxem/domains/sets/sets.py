"""
Set theory encoder.

Embeds finite sets over a bounded universe using characteristic vectors.
Set operations map to bitwise operations. Supports universes up to 64 elements.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


# =============================================================================
# Set Types
# =============================================================================

class SetType(Enum):
    """Types of set expressions."""
    EXPLICIT = auto()      # Explicitly enumerated set {1, 2, 3}
    EMPTY = auto()         # Empty set {}
    UNIVERSAL = auto()     # Universal set U (relative to universe)
    UNION = auto()         # A ∪ B
    INTERSECTION = auto()  # A ∩ B
    DIFFERENCE = auto()    # A \ B
    COMPLEMENT = auto()    # A^c (complement)
    SYMMETRIC_DIFF = auto() # A △ B
    POWER_SET = auto()     # P(A)
    CARTESIAN = auto()     # A × B


@dataclass(frozen=True)
class FiniteSet:
    """
    A finite set represented as a frozenset with optional universe context.

    For embedding, elements are mapped to indices 0-63 in the universe.
    Named elements use a hash-based mapping.

    Examples:
        FiniteSet({1, 2, 3})
        FiniteSet({'a', 'b', 'c'})
        FiniteSet(frozenset([1, 2, 3]))
    """
    elements: FrozenSet[Any] = field(default_factory=frozenset)
    universe: Optional[FrozenSet[Any]] = None  # If None, inferred from elements

    def __init__(self, elements: Iterable[Any] = (), universe: Optional[Iterable[Any]] = None):
        object.__setattr__(self, 'elements', frozenset(elements))
        if universe is not None:
            object.__setattr__(self, 'universe', frozenset(universe))
        else:
            object.__setattr__(self, 'universe', None)

    @classmethod
    def empty(cls) -> 'FiniteSet':
        """Create the empty set."""
        return cls(frozenset())

    @classmethod
    def from_range(cls, start: int, end: int) -> 'FiniteSet':
        """Create a set from integer range [start, end)."""
        return cls(frozenset(range(start, end)))

    def __contains__(self, element: Any) -> bool:
        """Check membership: element ∈ self."""
        return element in self.elements

    def __len__(self) -> int:
        """Cardinality: |self|."""
        return len(self.elements)

    def __iter__(self):
        """Iterate over elements."""
        return iter(self.elements)

    def __le__(self, other: 'FiniteSet') -> bool:
        """Subset: self ⊆ other."""
        return self.elements <= other.elements

    def __lt__(self, other: 'FiniteSet') -> bool:
        """Proper subset: self ⊂ other."""
        return self.elements < other.elements

    def __ge__(self, other: 'FiniteSet') -> bool:
        """Superset: self ⊇ other."""
        return self.elements >= other.elements

    def __gt__(self, other: 'FiniteSet') -> bool:
        """Proper superset: self ⊃ other."""
        return self.elements > other.elements

    def __or__(self, other: 'FiniteSet') -> 'FiniteSet':
        """Union: self ∪ other."""
        return FiniteSet(self.elements | other.elements)

    def __and__(self, other: 'FiniteSet') -> 'FiniteSet':
        """Intersection: self ∩ other."""
        return FiniteSet(self.elements & other.elements)

    def __sub__(self, other: 'FiniteSet') -> 'FiniteSet':
        """Difference: self \\ other."""
        return FiniteSet(self.elements - other.elements)

    def __xor__(self, other: 'FiniteSet') -> 'FiniteSet':
        """Symmetric difference: self △ other."""
        return FiniteSet(self.elements ^ other.elements)

    def complement(self, universe: Optional['FiniteSet'] = None) -> 'FiniteSet':
        """Complement relative to universe: U \\ self."""
        if universe is None:
            if self.universe is not None:
                univ = self.universe
            else:
                raise ValueError("Universe required for complement")
        else:
            univ = universe.elements
        return FiniteSet(univ - self.elements, universe=univ)

    def power_set(self) -> 'Set[FiniteSet]':
        """Power set: P(self). Warning: exponential size!"""
        from itertools import combinations
        elements = list(self.elements)
        result = set()
        for r in range(len(elements) + 1):
            for combo in combinations(elements, r):
                result.add(FiniteSet(frozenset(combo)))
        return result

    def cartesian_product(self, other: 'FiniteSet') -> 'FiniteSet':
        """Cartesian product: self × other."""
        pairs = frozenset((a, b) for a in self.elements for b in other.elements)
        return FiniteSet(pairs)

    def is_empty(self) -> bool:
        """Check if set is empty."""
        return len(self.elements) == 0

    def is_disjoint(self, other: 'FiniteSet') -> bool:
        """Check if sets are disjoint: self ∩ other = ∅."""
        return self.elements.isdisjoint(other.elements)

    def __repr__(self) -> str:
        if not self.elements:
            return "∅"
        # Sort for consistent display
        try:
            sorted_elems = sorted(self.elements)
        except TypeError:
            sorted_elems = list(self.elements)
        return "{" + ", ".join(str(e) for e in sorted_elems) + "}"

    def __hash__(self) -> int:
        return hash(self.elements)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FiniteSet):
            return False
        return self.elements == other.elements


# =============================================================================
# Embedding Layout
# =============================================================================

# Embedding layout for sets (absolute positions in 128-dim embedding):
# dims 0-7:   Domain tag
# dims 8-15:  Set type one-hot [EXPLICIT, EMPTY, UNIVERSAL, UNION, INTER, DIFF, COMP, SYM]
# dims 16-71: Element bitmap (56 bits for elements 0-55)
# dims 72-75: Metadata (cardinality, universe size, flags)
# dims 76-95: Reserved
# dims 96-127: Cross-domain composition

# Absolute positions
TYPE_START = 8
BITMAP_START = 16
BITMAP_SIZE = 56  # Support up to 56 elements
CARDINALITY_POS = 72
UNIVERSE_SIZE_POS = 73
EMPTY_FLAG_POS = 74
UNIVERSAL_FLAG_POS = 75


# =============================================================================
# Set Encoder
# =============================================================================

class SetEncoder:
    """
    Encoder for finite sets.

    All set operations are - Union, intersection, difference via bitwise operations
    - Membership and subset checking via bit comparison
    - Cardinality via popcount

    Supports sets over a finite universe of up to 64 elements.
    Elements are mapped to indices using:
    - Integers 0-63 map directly
    - Other values use hash % 64
    """

    domain_tag = DOMAIN_TAGS["set_finite"]
    domain_name = "set_finite"

    def __init__(self, universe_size: int = 56):
        """
        Initialize encoder with universe size.

        Args:
            universe_size: Maximum number of elements (up to 56)
        """
        self.universe_size = min(universe_size, BITMAP_SIZE)
        self._element_map: Dict[Any, int] = {}
        self._reverse_map: Dict[int, Any] = {}
        self._next_idx = 0

    def _element_to_index(self, element: Any) -> int:
        """Map an element to a bit index (0 to BITMAP_SIZE-1)."""
        # Check if already mapped
        if element in self._element_map:
            return self._element_map[element]

        # Integers in range map directly
        if isinstance(element, int) and 0 <= element < BITMAP_SIZE:
            self._element_map[element] = element
            self._reverse_map[element] = element
            return element

        # Assign next available index
        if self._next_idx < BITMAP_SIZE:
            idx = self._next_idx
            # Skip indices used by direct integer mapping
            while idx in self._reverse_map and idx < BITMAP_SIZE:
                idx += 1
            if idx < BITMAP_SIZE:
                self._element_map[element] = idx
                self._reverse_map[idx] = element
                self._next_idx = idx + 1
                return idx

        # Fallback: hash-based mapping
        return hash(element) % BITMAP_SIZE

    def encode(self, s: Union[FiniteSet, Set, FrozenSet, Iterable]) -> Any:
        """
        Encode a finite set.

        Args:
            s: A FiniteSet, set, frozenset, or iterable of elements

        Returns:
            128-dim embedding with set representation
        """
        backend = get_backend()
        # Normalize input
        if isinstance(s, FiniteSet):
            elements = s.elements
        elif isinstance(s, (set, frozenset)):
            elements = s
        else:
            elements = frozenset(s)

        emb = create_embedding()

        # Domain tag (dims 0-7)
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Set type (dims 8-15)
        if not elements:
            # Empty set
            emb = backend.at_add(emb, TYPE_START + 1, 1.0)  # EMPTY type
            emb = backend.at_add(emb, EMPTY_FLAG_POS, 1.0)
        else:
            # Explicit set
            emb = backend.at_add(emb, TYPE_START + 0, 1.0)  # EXPLICIT type

        # Element bitmap (dims 16-71)
        for elem in elements:
            idx = self._element_to_index(elem)
            if 0 <= idx < BITMAP_SIZE:
                emb = backend.at_add(emb, BITMAP_START + idx, 1.0)

        # Cardinality (log scale)
        cardinality = len(elements)
        emb = backend.at_add(emb, CARDINALITY_POS, 
            backend.log(backend.array(float(cardinality + 1)))
        )

        # Universe size
        emb = backend.at_add(emb, UNIVERSE_SIZE_POS, 
            backend.log(backend.array(float(self.universe_size + 1)))
        )

        return emb

    def decode(self, emb: Any) -> FiniteSet:
        """
        Decode embedding back to a FiniteSet.

        Uses the element bitmap to reconstruct the set.
        """
        elements = set()

        # Read bitmap
        for i in range(BITMAP_SIZE):
            if emb[BITMAP_START + i].item() > 0.5:
                # Try to get original element from reverse map
                if i in self._reverse_map:
                    elements.add(self._reverse_map[i])
                else:
                    elements.add(i)

        return FiniteSet(elements)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid set."""
        backend = get_backend()
        # Check for set_finite tag pattern
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Set operations
    # =========================================================================

    def union(self, emb1: Any, emb2: Any) -> Any:
        """
        Set union: A ∪ B

        Bitwise OR on element bitmaps.
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Union type
        result = backend.at_add(result, TYPE_START + 3, 1.0)  # UNION type

        # Bitmap: OR
        for i in range(BITMAP_SIZE):
            val1 = emb1[BITMAP_START + i].item()
            val2 = emb2[BITMAP_START + i].item()
            if val1 > 0.5 or val2 > 0.5:
                result = backend.at_add(result, BITMAP_START + i, 1.0)

        # Recompute cardinality
        card = sum(1 for i in range(BITMAP_SIZE) if result[BITMAP_START + i].item() > 0.5)
        result = backend.at_add(result, CARDINALITY_POS, backend.log(backend.array(float(card + 1))))

        # Empty flag
        if card == 0:
            result = backend.at_add(result, EMPTY_FLAG_POS, 1.0)

        return result

    def intersection(self, emb1: Any, emb2: Any) -> Any:
        """
        Set intersection: A ∩ B

        Bitwise AND on element bitmaps.
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Intersection type
        result = backend.at_add(result, TYPE_START + 4, 1.0)  # INTERSECTION type

        # Bitmap: AND
        for i in range(BITMAP_SIZE):
            val1 = emb1[BITMAP_START + i].item()
            val2 = emb2[BITMAP_START + i].item()
            if val1 > 0.5 and val2 > 0.5:
                result = backend.at_add(result, BITMAP_START + i, 1.0)

        # Recompute cardinality
        card = sum(1 for i in range(BITMAP_SIZE) if result[BITMAP_START + i].item() > 0.5)
        result = backend.at_add(result, CARDINALITY_POS, backend.log(backend.array(float(card + 1))))

        # Empty flag
        if card == 0:
            result = backend.at_add(result, EMPTY_FLAG_POS, 1.0)

        return result

    def difference(self, emb1: Any, emb2: Any) -> Any:
        """
        Set difference: A \\ B (elements in A but not in B)

        A AND (NOT B) on bitmaps.
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Difference type
        result = backend.at_add(result, TYPE_START + 5, 1.0)  # DIFFERENCE type

        # Bitmap: A AND NOT B
        for i in range(BITMAP_SIZE):
            val1 = emb1[BITMAP_START + i].item()
            val2 = emb2[BITMAP_START + i].item()
            if val1 > 0.5 and val2 <= 0.5:
                result = backend.at_add(result, BITMAP_START + i, 1.0)

        # Recompute cardinality
        card = sum(1 for i in range(BITMAP_SIZE) if result[BITMAP_START + i].item() > 0.5)
        result = backend.at_add(result, CARDINALITY_POS, backend.log(backend.array(float(card + 1))))

        # Empty flag
        if card == 0:
            result = backend.at_add(result, EMPTY_FLAG_POS, 1.0)

        return result

    def symmetric_difference(self, emb1: Any, emb2: Any) -> Any:
        """
        Symmetric difference: A △ B (elements in exactly one of A or B)

        XOR on bitmaps.
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Symmetric difference type
        result = backend.at_add(result, TYPE_START + 7, 1.0)  # SYMMETRIC_DIFF type

        # Bitmap: XOR
        for i in range(BITMAP_SIZE):
            val1 = emb1[BITMAP_START + i].item()
            val2 = emb2[BITMAP_START + i].item()
            in1 = val1 > 0.5
            in2 = val2 > 0.5
            if in1 != in2:  # XOR
                result = backend.at_add(result, BITMAP_START + i, 1.0)

        # Recompute cardinality
        card = sum(1 for i in range(BITMAP_SIZE) if result[BITMAP_START + i].item() > 0.5)
        result = backend.at_add(result, CARDINALITY_POS, backend.log(backend.array(float(card + 1))))

        # Empty flag
        if card == 0:
            result = backend.at_add(result, EMPTY_FLAG_POS, 1.0)

        return result

    def complement(self, emb: Any, universe_size: Optional[int] = None) -> Any:
        """
        Set complement: A^c (elements not in A, within universe)

        Bitwise NOT (within universe bounds).
        """
        backend = get_backend()
        if universe_size is None:
            universe_size = self.universe_size

        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Complement type
        result = backend.at_add(result, TYPE_START + 6, 1.0)  # COMPLEMENT type

        # Bitmap: NOT (within universe)
        for i in range(min(universe_size, BITMAP_SIZE)):
            val = emb[BITMAP_START + i].item()
            if val <= 0.5:
                result = backend.at_add(result, BITMAP_START + i, 1.0)

        # Recompute cardinality
        card = sum(1 for i in range(BITMAP_SIZE) if result[BITMAP_START + i].item() > 0.5)
        result = backend.at_add(result, CARDINALITY_POS, backend.log(backend.array(float(card + 1))))

        # Empty flag
        if card == 0:
            result = backend.at_add(result, EMPTY_FLAG_POS, 1.0)

        # Universal flag (if original was empty)
        if emb[EMPTY_FLAG_POS].item() > 0.5:
            result = backend.at_add(result, UNIVERSAL_FLAG_POS, 1.0)

        return result

    # =========================================================================
    # Predicates
    # =========================================================================

    def is_subset(self, emb1: Any, emb2: Any) -> bool:
        """
        Check subset: A ⊆ B

        A ⊆ B iff A ∩ B = A iff (for all i: A[i] → B[i])
        """
        for i in range(BITMAP_SIZE):
            in_a = emb1[BITMAP_START + i].item() > 0.5
            in_b = emb2[BITMAP_START + i].item() > 0.5
            if in_a and not in_b:
                return False
        return True

    def is_proper_subset(self, emb1: Any, emb2: Any) -> bool:
        """
        Check proper subset: A ⊂ B

        A ⊂ B iff A ⊆ B and A ≠ B
        """
        if not self.is_subset(emb1, emb2):
            return False
        return not self.equals(emb1, emb2)

    def is_superset(self, emb1: Any, emb2: Any) -> bool:
        """
        Check superset: A ⊇ B

        A ⊇ B iff B ⊆ A
        """
        return self.is_subset(emb2, emb1)

    def equals(self, emb1: Any, emb2: Any) -> bool:
        """
        Check set equality: A = B

        Compare bitmaps.
        """
        for i in range(BITMAP_SIZE):
            in1 = emb1[BITMAP_START + i].item() > 0.5
            in2 = emb2[BITMAP_START + i].item() > 0.5
            if in1 != in2:
                return False
        return True

    def is_disjoint(self, emb1: Any, emb2: Any) -> bool:
        """
        Check disjointness: A ∩ B = ∅

        No overlapping bits.
        """
        for i in range(BITMAP_SIZE):
            in1 = emb1[BITMAP_START + i].item() > 0.5
            in2 = emb2[BITMAP_START + i].item() > 0.5
            if in1 and in2:
                return False
        return True

    def contains(self, emb: Any, element: Any) -> bool:
        """
        Check membership: element ∈ A

        Check bit at element's index.
        """
        idx = self._element_to_index(element)
        if 0 <= idx < BITMAP_SIZE:
            return emb[BITMAP_START + idx].item() > 0.5
        return False

    def is_empty(self, emb: Any) -> bool:
        """
        Check if set is empty: A = ∅

        Check empty flag or all bits zero.
        """
        if emb[EMPTY_FLAG_POS].item() > 0.5:
            return True
        return self.cardinality(emb) == 0

    def cardinality(self, emb: Any) -> int:
        """
        Get set cardinality: |A|

        Popcount of bitmap.
        """
        count = 0
        for i in range(BITMAP_SIZE):
            if emb[BITMAP_START + i].item() > 0.5:
                count += 1
        return count

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def encode_empty(self) -> Any:
        """Encode the empty set ∅."""
        return self.encode(FiniteSet.empty())

    def encode_singleton(self, element: Any) -> Any:
        """Encode a singleton set {element}."""
        return self.encode(FiniteSet({element}))

    def encode_range(self, start: int, end: int) -> Any:
        """Encode a set from integer range [start, end)."""
        return self.encode(FiniteSet.from_range(start, end))

    def encode_universal(self, size: Optional[int] = None) -> Any:
        """Encode the universal set U = {0, 1, ..., size-1}."""
        if size is None:
            size = self.universe_size
        return self.encode(FiniteSet.from_range(0, size))

    def get_elements(self, emb: Any) -> Set[Any]:
        """Extract elements from embedding."""
        return self.decode(emb).elements
