"""
Binary Relation Encoder.

A binary relation R on a set A is a subset of A × A.
Relations can be represented as adjacency matrices (for small sets).

Properties that can be checked deterministically:
- Reflexive: ∀x: (x,x) ∈ R
- Symmetric: ∀x,y: (x,y) ∈ R → (y,x) ∈ R
- Antisymmetric: ∀x,y: (x,y) ∈ R ∧ (y,x) ∈ R → x = y
- Transitive: ∀x,y,z: (x,y) ∈ R ∧ (y,z) ∈ R → (x,z) ∈ R
- Total: ∀x,y: (x,y) ∈ R ∨ (y,x) ∈ R

Operations:
- Composition: R ∘ S = {(a,c) : ∃b: (a,b) ∈ R ∧ (b,c) ∈ S}
- Inverse: R⁻¹ = {(b,a) : (a,b) ∈ R}
- Closure: reflexive, symmetric, transitive closures

Supports relations on sets of up to 8 elements (64-bit adjacency matrix).
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
)

# Get the proper domain tag
SET_RELATION_TAG = DOMAIN_TAGS["set_relation"]


# =============================================================================
# Relation Types
# =============================================================================

class RelationType(Enum):
    """Types of relations based on properties."""
    GENERAL = auto()        # No special properties
    EQUIVALENCE = auto()    # Reflexive, symmetric, transitive
    PARTIAL_ORDER = auto()  # Reflexive, antisymmetric, transitive
    TOTAL_ORDER = auto()    # Partial order + total
    PREORDER = auto()       # Reflexive, transitive
    STRICT_ORDER = auto()   # Irreflexive, asymmetric, transitive
    FUNCTION = auto()       # Each element maps to at most one
    BIJECTION = auto()      # Function that is one-to-one and onto


@dataclass(frozen=True)
class Relation:
    """
    A binary relation as a set of ordered pairs.

    Examples:
        Relation({(1,2), (2,3), (1,3)})  # Explicit pairs
        Relation.identity(3)  # {(0,0), (1,1), (2,2)}
        Relation.less_than(3)  # {(0,1), (0,2), (1,2)}
    """
    pairs: FrozenSet[Tuple[Any, Any]]
    domain_set: Optional[FrozenSet[Any]] = None  # Domain if known
    codomain_set: Optional[FrozenSet[Any]] = None  # Codomain if known

    def __init__(
        self,
        pairs: Set[Tuple[Any, Any]] = None,
        domain_set: Optional[Set[Any]] = None,
        codomain_set: Optional[Set[Any]] = None
    ):
        if pairs is None:
            pairs = frozenset()
        object.__setattr__(self, 'pairs', frozenset(pairs))
        if domain_set is not None:
            object.__setattr__(self, 'domain_set', frozenset(domain_set))
        else:
            object.__setattr__(self, 'domain_set', None)
        if codomain_set is not None:
            object.__setattr__(self, 'codomain_set', frozenset(codomain_set))
        else:
            object.__setattr__(self, 'codomain_set', None)

    @classmethod
    def empty(cls) -> 'Relation':
        """Create the empty relation."""
        return cls(frozenset())

    @classmethod
    def identity(cls, n: int) -> 'Relation':
        """Create the identity relation on {0, 1, ..., n-1}."""
        pairs = frozenset((i, i) for i in range(n))
        return cls(pairs)

    @classmethod
    def universal(cls, n: int) -> 'Relation':
        """Create the universal relation (all pairs) on {0, 1, ..., n-1}."""
        pairs = frozenset((i, j) for i in range(n) for j in range(n))
        return cls(pairs)

    @classmethod
    def less_than(cls, n: int) -> 'Relation':
        """Create the < relation on {0, 1, ..., n-1}."""
        pairs = frozenset((i, j) for i in range(n) for j in range(n) if i < j)
        return cls(pairs)

    @classmethod
    def less_equal(cls, n: int) -> 'Relation':
        """Create the ≤ relation on {0, 1, ..., n-1}."""
        pairs = frozenset((i, j) for i in range(n) for j in range(n) if i <= j)
        return cls(pairs)

    @classmethod
    def divides(cls, n: int) -> 'Relation':
        """Create the divides relation on {1, 2, ..., n}."""
        pairs = frozenset(
            (i, j) for i in range(1, n + 1) for j in range(1, n + 1)
            if j % i == 0
        )
        return cls(pairs)

    def domain(self) -> FrozenSet[Any]:
        """Get the domain (set of first elements)."""
        return frozenset(a for a, b in self.pairs)

    def codomain(self) -> FrozenSet[Any]:
        """Get the codomain/range (set of second elements)."""
        return frozenset(b for a, b in self.pairs)

    def field(self) -> FrozenSet[Any]:
        """Get the field (domain ∪ codomain)."""
        return self.domain() | self.codomain()

    def __contains__(self, pair: Tuple[Any, Any]) -> bool:
        """Check if pair is in relation."""
        return pair in self.pairs

    def __len__(self) -> int:
        """Number of pairs."""
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    # =========================================================================
    # Relation Operations
    # =========================================================================

    def inverse(self) -> 'Relation':
        """
        Inverse relation: R⁻¹ = {(b,a) : (a,b) ∈ R}

        """
        return Relation(frozenset((b, a) for a, b in self.pairs))

    def compose(self, other: 'Relation') -> 'Relation':
        """
        Composition: self ∘ other = {(a,c) : ∃b: (a,b) ∈ other ∧ (b,c) ∈ self}

        Note: Composition order follows function notation (self after other).

        """
        result = set()
        for a, b1 in other.pairs:
            for b2, c in self.pairs:
                if b1 == b2:
                    result.add((a, c))
        return Relation(frozenset(result))

    def union(self, other: 'Relation') -> 'Relation':
        """Union of relations."""
        return Relation(self.pairs | other.pairs)

    def intersection(self, other: 'Relation') -> 'Relation':
        """Intersection of relations."""
        return Relation(self.pairs & other.pairs)

    def difference(self, other: 'Relation') -> 'Relation':
        """Difference of relations."""
        return Relation(self.pairs - other.pairs)

    # =========================================================================
    # Properties
    # =========================================================================

    def is_reflexive(self, universe: Optional[Set[Any]] = None) -> bool:
        """
        Check reflexivity: ∀x ∈ U: (x,x) ∈ R

        """
        if universe is None:
            universe = self.field()
        return all((x, x) in self.pairs for x in universe)

    def is_irreflexive(self) -> bool:
        """
        Check irreflexivity: ∀x: (x,x) ∉ R

        """
        return all(a != b for a, b in self.pairs)

    def is_symmetric(self) -> bool:
        """
        Check symmetry: ∀x,y: (x,y) ∈ R → (y,x) ∈ R

        """
        return all((b, a) in self.pairs for a, b in self.pairs)

    def is_antisymmetric(self) -> bool:
        """
        Check antisymmetry: ∀x,y: (x,y) ∈ R ∧ (y,x) ∈ R → x = y

        """
        for a, b in self.pairs:
            if a != b and (b, a) in self.pairs:
                return False
        return True

    def is_asymmetric(self) -> bool:
        """
        Check asymmetry: ∀x,y: (x,y) ∈ R → (y,x) ∉ R

        """
        return all((b, a) not in self.pairs for a, b in self.pairs)

    def is_transitive(self) -> bool:
        """
        Check transitivity: ∀x,y,z: (x,y) ∈ R ∧ (y,z) ∈ R → (x,z) ∈ R

        """
        for a, b in self.pairs:
            for c, d in self.pairs:
                if b == c and (a, d) not in self.pairs:
                    return False
        return True

    def is_total(self, universe: Optional[Set[Any]] = None) -> bool:
        """
        Check totality: ∀x,y ∈ U: (x,y) ∈ R ∨ (y,x) ∈ R

        """
        if universe is None:
            universe = self.field()
        for x in universe:
            for y in universe:
                if (x, y) not in self.pairs and (y, x) not in self.pairs:
                    return False
        return True

    def is_equivalence(self, universe: Optional[Set[Any]] = None) -> bool:
        """Check if relation is an equivalence (reflexive, symmetric, transitive)."""
        return (self.is_reflexive(universe) and
                self.is_symmetric() and
                self.is_transitive())

    def is_partial_order(self, universe: Optional[Set[Any]] = None) -> bool:
        """Check if relation is a partial order (reflexive, antisymmetric, transitive)."""
        return (self.is_reflexive(universe) and
                self.is_antisymmetric() and
                self.is_transitive())

    def is_total_order(self, universe: Optional[Set[Any]] = None) -> bool:
        """Check if relation is a total order (partial order + total)."""
        return self.is_partial_order(universe) and self.is_total(universe)

    def is_strict_order(self) -> bool:
        """Check if relation is a strict order (irreflexive, asymmetric, transitive)."""
        return (self.is_irreflexive() and
                self.is_asymmetric() and
                self.is_transitive())

    def is_function(self) -> bool:
        """Check if relation is a function (each input maps to at most one output)."""
        seen_inputs = set()
        for a, b in self.pairs:
            if a in seen_inputs:
                # Check if same output
                for a2, b2 in self.pairs:
                    if a2 == a and b2 != b:
                        return False
            seen_inputs.add(a)
        return True

    # =========================================================================
    # Closures
    # =========================================================================

    def reflexive_closure(self, universe: Optional[Set[Any]] = None) -> 'Relation':
        """
        Reflexive closure: smallest reflexive relation containing R.

        R ∪ {(x,x) : x ∈ U}
        """
        if universe is None:
            universe = self.field()
        extra = frozenset((x, x) for x in universe)
        return Relation(self.pairs | extra)

    def symmetric_closure(self) -> 'Relation':
        """
        Symmetric closure: smallest symmetric relation containing R.

        R ∪ R⁻¹
        """
        inverse_pairs = frozenset((b, a) for a, b in self.pairs)
        return Relation(self.pairs | inverse_pairs)

    def transitive_closure(self) -> 'Relation':
        """
        Transitive closure: smallest transitive relation containing R.

        Uses Warshall's algorithm for finite relations.
        """
        # Convert to adjacency matrix approach
        elements = sorted(self.field())
        n = len(elements)
        if n == 0:
            return Relation(frozenset())

        elem_to_idx = {e: i for i, e in enumerate(elements)}

        # Build adjacency matrix
        matrix = [[False] * n for _ in range(n)]
        for a, b in self.pairs:
            matrix[elem_to_idx[a]][elem_to_idx[b]] = True

        # Warshall's algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if matrix[i][k] and matrix[k][j]:
                        matrix[i][j] = True

        # Convert back to pairs
        result = frozenset(
            (elements[i], elements[j])
            for i in range(n) for j in range(n)
            if matrix[i][j]
        )
        return Relation(result)

    def equivalence_classes(self) -> List[FrozenSet[Any]]:
        """
        Get equivalence classes (if this is an equivalence relation).

        Returns list of equivalence classes (disjoint sets).
        """
        if not self.is_equivalence():
            return []

        elements = self.field()
        classes = []
        seen = set()

        for x in elements:
            if x in seen:
                continue
            # Find all elements equivalent to x
            eq_class = frozenset(y for y in elements if (x, y) in self.pairs)
            classes.append(eq_class)
            seen.update(eq_class)

        return classes

    def __repr__(self) -> str:
        if not self.pairs:
            return "∅"
        pairs_str = ", ".join(f"({a},{b})" for a, b in sorted(self.pairs))
        return "{" + pairs_str + "}"


# =============================================================================
# Embedding Layout for Relations
# =============================================================================

# For relations on sets of up to 8 elements, we use a 64-bit adjacency matrix.
# Absolute positions in 128-dim embedding:
# dims 0-7:   Domain tag
# dims 8-71:  Adjacency matrix (8×8 = 64 bits)
# dims 72-79: Relation type one-hot (8 types)
# dim 80:     Number of pairs (log scale)
# dim 81:     Domain size
# dims 82-87: Property flags [reflexive, symmetric, antisym, transitive, total, function]

ADJACENCY_START = 8
ADJACENCY_SIZE = 64  # 8x8 matrix
TYPE_START = 72
PAIR_COUNT_POS = 80
DOMAIN_SIZE_POS = 81
PROPERTY_FLAGS_START = 82


# =============================================================================
# Relation Encoder
# =============================================================================

class RelationEncoder:
    """
    Encoder for binary relations.

    Represents relations as adjacency matrices for sets of up to 8 elements.
    All operations are deterministic.
    """

    domain_tag = SET_RELATION_TAG
    domain_name = "set_relation"

    def __init__(self, max_elements: int = 8):
        """
        Initialize encoder.

        Args:
            max_elements: Maximum elements in domain (up to 8)
        """
        self.max_elements = min(max_elements, 8)
        self._element_map: Dict[Any, int] = {}
        self._reverse_map: Dict[int, Any] = {}

    def _element_to_index(self, element: Any) -> int:
        """Map element to index 0-7."""
        if element in self._element_map:
            return self._element_map[element]

        if isinstance(element, int) and 0 <= element < 8:
            self._element_map[element] = element
            self._reverse_map[element] = element
            return element

        idx = len(self._element_map)
        if idx < 8:
            self._element_map[element] = idx
            self._reverse_map[idx] = element
            return idx

        return hash(element) % 8

    def _pair_to_bit(self, a: int, b: int) -> int:
        """Convert (row, col) to bit index in 64-bit adjacency matrix."""
        return a * 8 + b

    def encode(self, relation: Union[Relation, Set[Tuple[Any, Any]]]) -> Any:
        """
        Encode a binary relation.

        Args:
            relation: A Relation or set of (a, b) pairs

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(relation, set):
            relation = Relation(relation)

        emb = create_embedding()

        # Domain tag (dims 0-7)
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Adjacency matrix (dims 8-71)
        for a, b in relation.pairs:
            ai = self._element_to_index(a)
            bi = self._element_to_index(b)
            bit_idx = self._pair_to_bit(ai, bi)
            if 0 <= bit_idx < ADJACENCY_SIZE:
                emb = backend.at_add(emb, ADJACENCY_START + bit_idx, 1.0)

        # Determine relation type (dims 72-79)
        if relation.is_equivalence():
            type_idx = 1  # EQUIVALENCE
        elif relation.is_total_order():
            type_idx = 3  # TOTAL_ORDER
        elif relation.is_partial_order():
            type_idx = 2  # PARTIAL_ORDER
        elif relation.is_strict_order():
            type_idx = 5  # STRICT_ORDER
        elif relation.is_function():
            type_idx = 6  # FUNCTION
        else:
            type_idx = 0  # GENERAL

        emb = backend.at_add(emb, TYPE_START + type_idx, 1.0)

        # Pair count
        emb = backend.at_add(emb, PAIR_COUNT_POS, 
            backend.log(backend.array(float(len(relation.pairs) + 1)))
        )

        # Domain size
        domain_size = len(relation.field())
        emb = backend.at_add(emb, DOMAIN_SIZE_POS, float(domain_size))

        # Property flags
        universe = relation.field()
        props = [
            relation.is_reflexive(universe),
            relation.is_symmetric(),
            relation.is_antisymmetric(),
            relation.is_transitive(),
            relation.is_total(universe),
            relation.is_function(),
        ]
        for i, prop in enumerate(props):
            if prop:
                emb = backend.at_add(emb, PROPERTY_FLAGS_START + i, 1.0)

        return emb

    def decode(self, emb: Any) -> Relation:
        """Decode embedding to a Relation."""
        pairs = set()

        for bit_idx in range(ADJACENCY_SIZE):
            if emb[ADJACENCY_START + bit_idx].item() > 0.5:
                a = bit_idx // 8
                b = bit_idx % 8
                # Try to get original elements
                a_elem = self._reverse_map.get(a, a)
                b_elem = self._reverse_map.get(b, b)
                pairs.add((a_elem, b_elem))

        return Relation(pairs)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid relation."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Operations
    # =========================================================================

    def compose(self, emb1: Any, emb2: Any) -> Any:
        """
        Relation composition: R₁ ∘ R₂

        Matrix multiplication (over Boolean semiring)
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Compute composition via matrix multiplication
        for i in range(8):
            for j in range(8):
                # Check if (i,j) should be in composition
                for k in range(8):
                    bit_ik = emb2[ADJACENCY_START + i * 8 + k].item() > 0.5
                    bit_kj = emb1[ADJACENCY_START + k * 8 + j].item() > 0.5
                    if bit_ik and bit_kj:
                        result = backend.at_add(result, ADJACENCY_START + i * 8 + j, 1.0)
                        break

        return result

    def inverse(self, emb: Any) -> Any:
        """
        Inverse relation: R⁻¹

        Transpose adjacency matrix
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        for i in range(8):
            for j in range(8):
                if emb[ADJACENCY_START + i * 8 + j].item() > 0.5:
                    result = backend.at_add(result, ADJACENCY_START + j * 8 + i, 1.0)

        return result

    def union(self, emb1: Any, emb2: Any) -> Any:
        """Union of relations: R₁ ∪ R₂"""
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        for bit_idx in range(ADJACENCY_SIZE):
            if (emb1[ADJACENCY_START + bit_idx].item() > 0.5 or
                emb2[ADJACENCY_START + bit_idx].item() > 0.5):
                result = backend.at_add(result, ADJACENCY_START + bit_idx, 1.0)

        return result

    def intersection(self, emb1: Any, emb2: Any) -> Any:
        """Intersection of relations: R₁ ∩ R₂"""
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        for bit_idx in range(ADJACENCY_SIZE):
            if (emb1[ADJACENCY_START + bit_idx].item() > 0.5 and
                emb2[ADJACENCY_START + bit_idx].item() > 0.5):
                result = backend.at_add(result, ADJACENCY_START + bit_idx, 1.0)

        return result

    # =========================================================================
    # Property checks
    # =========================================================================

    def is_reflexive(self, emb: Any, n: int = 8) -> bool:
        """Check reflexivity from embedding."""
        for i in range(n):
            if emb[ADJACENCY_START + i * 8 + i].item() <= 0.5:
                return False
        return True

    def is_symmetric(self, emb: Any) -> bool:
        """Check symmetry from embedding."""
        for i in range(8):
            for j in range(8):
                ij = emb[ADJACENCY_START + i * 8 + j].item() > 0.5
                ji = emb[ADJACENCY_START + j * 8 + i].item() > 0.5
                if ij and not ji:
                    return False
        return True

    def is_transitive(self, emb: Any) -> bool:
        """Check transitivity from embedding."""
        for i in range(8):
            for j in range(8):
                if emb[ADJACENCY_START + i * 8 + j].item() <= 0.5:
                    continue
                for k in range(8):
                    if emb[ADJACENCY_START + j * 8 + k].item() > 0.5:
                        if emb[ADJACENCY_START + i * 8 + k].item() <= 0.5:
                            return False
        return True

    def is_antisymmetric(self, emb: Any) -> bool:
        """Check antisymmetry from embedding."""
        for i in range(8):
            for j in range(8):
                if i != j:
                    ij = emb[ADJACENCY_START + i * 8 + j].item() > 0.5
                    ji = emb[ADJACENCY_START + j * 8 + i].item() > 0.5
                    if ij and ji:
                        return False
        return True

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def encode_identity(self, n: int) -> Any:
        """Encode identity relation."""
        return self.encode(Relation.identity(n))

    def encode_less_than(self, n: int) -> Any:
        """Encode < relation."""
        return self.encode(Relation.less_than(n))

    def encode_less_equal(self, n: int) -> Any:
        """Encode ≤ relation."""
        return self.encode(Relation.less_equal(n))
