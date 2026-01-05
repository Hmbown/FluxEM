"""
Mathematical Function Encoder.

A function f: A → B is a relation where each element of A maps to exactly
one element of B. Functions can be:

- Injective (one-to-one): f(x) = f(y) → x = y
- Surjective (onto): ∀y ∈ B: ∃x ∈ A: f(x) = y
- Bijective: both injective and surjective

Operations:
- Composition: (g ∘ f)(x) = g(f(x))
- Inverse: f⁻¹ (only for bijections)
- Image: f(S) = {f(x) : x ∈ S}
- Preimage: f⁻¹(T) = {x : f(x) ∈ T}

All operations are deterministic for finite functions.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
)

# Get the proper domain tag
SET_FUNCTION_TAG = DOMAIN_TAGS["set_function"]


# =============================================================================
# Function Types
# =============================================================================

class FunctionType(Enum):
    """Types of functions based on properties."""
    GENERAL = auto()      # No special properties
    INJECTIVE = auto()    # One-to-one
    SURJECTIVE = auto()   # Onto
    BIJECTIVE = auto()    # Both injective and surjective
    CONSTANT = auto()     # f(x) = c for all x
    IDENTITY = auto()     # f(x) = x
    PERMUTATION = auto()  # Bijection from set to itself


@dataclass(frozen=True)
class Function:
    """
    A mathematical function as a mapping from domain to codomain.

    Internally represented as a dict for efficient lookup.

    Examples:
        Function({0: 'a', 1: 'b', 2: 'c'})
        Function.identity(5)
        Function.constant(range(5), 42)
    """
    mapping: Dict[Any, Any]  # Stored as regular dict, treated as immutable
    domain_set: Optional[FrozenSet[Any]] = None
    codomain_set: Optional[FrozenSet[Any]] = None

    def __init__(
        self,
        mapping: Dict[Any, Any] = None,
        domain_set: Optional[Set[Any]] = None,
        codomain_set: Optional[Set[Any]] = None
    ):
        if mapping is None:
            mapping = {}
        # Store as frozen (immutable) by convention
        object.__setattr__(self, 'mapping', dict(mapping))
        if domain_set is not None:
            object.__setattr__(self, 'domain_set', frozenset(domain_set))
        else:
            object.__setattr__(self, 'domain_set', frozenset(mapping.keys()))
        if codomain_set is not None:
            object.__setattr__(self, 'codomain_set', frozenset(codomain_set))
        else:
            object.__setattr__(self, 'codomain_set', frozenset(mapping.values()))

    @classmethod
    def identity(cls, n: int) -> 'Function':
        """Create identity function on {0, 1, ..., n-1}."""
        return cls({i: i for i in range(n)})

    @classmethod
    def constant(cls, domain: Set[Any], value: Any) -> 'Function':
        """Create constant function f(x) = value for all x in domain."""
        return cls({x: value for x in domain})

    @classmethod
    def from_list(cls, values: list) -> 'Function':
        """Create function f(i) = values[i]."""
        return cls({i: v for i, v in enumerate(values)})

    @classmethod
    def successor(cls, n: int, modular: bool = False) -> 'Function':
        """
        Create successor function on {0, ..., n-1}.
        If modular, wraps around (mod n).
        """
        if modular:
            return cls({i: (i + 1) % n for i in range(n)})
        else:
            return cls({i: i + 1 for i in range(n - 1)})

    def __call__(self, x: Any) -> Any:
        """Apply function: f(x)."""
        if x not in self.mapping:
            raise ValueError(f"Element {x} not in domain")
        return self.mapping[x]

    def __len__(self) -> int:
        """Size of domain."""
        return len(self.mapping)

    def domain(self) -> FrozenSet[Any]:
        """Get the domain."""
        return frozenset(self.mapping.keys())

    def codomain(self) -> FrozenSet[Any]:
        """Get the codomain (may be larger than range)."""
        if self.codomain_set is not None:
            return self.codomain_set
        return frozenset(self.mapping.values())

    def range(self) -> FrozenSet[Any]:
        """Get the range (actual image)."""
        return frozenset(self.mapping.values())

    def as_pairs(self) -> FrozenSet[Tuple[Any, Any]]:
        """Get as set of (input, output) pairs."""
        return frozenset(self.mapping.items())

    # =========================================================================
    # Properties
    # =========================================================================

    def is_injective(self) -> bool:
        """
        Check if function is injective (one-to-one).

        f(x) = f(y) → x = y
        """
        values = list(self.mapping.values())
        return len(values) == len(set(values))

    def is_surjective(self, codomain: Optional[Set[Any]] = None) -> bool:
        """
        Check if function is surjective (onto).

        Every element in codomain is hit.
        """
        if codomain is None:
            codomain = self.codomain_set if self.codomain_set else set(self.mapping.values())
        return set(self.mapping.values()) >= set(codomain)

    def is_bijective(self, codomain: Optional[Set[Any]] = None) -> bool:
        """Check if function is bijective (one-to-one and onto)."""
        return self.is_injective() and self.is_surjective(codomain)

    def is_constant(self) -> bool:
        """Check if function is constant."""
        values = set(self.mapping.values())
        return len(values) <= 1

    def is_identity(self) -> bool:
        """Check if function is identity."""
        return all(k == v for k, v in self.mapping.items())

    def is_permutation(self) -> bool:
        """Check if function is a permutation (bijection to same set)."""
        return (self.is_bijective() and
                set(self.mapping.keys()) == set(self.mapping.values()))

    def is_involution(self) -> bool:
        """Check if f(f(x)) = x for all x (self-inverse)."""
        for x, y in self.mapping.items():
            if y not in self.mapping or self.mapping[y] != x:
                return False
        return True

    # =========================================================================
    # Operations
    # =========================================================================

    def compose(self, other: 'Function') -> 'Function':
        """
        Composition: (self ∘ other)(x) = self(other(x))

        """
        result = {}
        for x, y in other.mapping.items():
            if y in self.mapping:
                result[x] = self.mapping[y]
        return Function(result)

    def inverse(self) -> Optional['Function']:
        """
        Inverse function: f⁻¹

        Only exists for bijective functions.
        """
        if not self.is_injective():
            return None  # No inverse exists
        return Function({v: k for k, v in self.mapping.items()})

    def restrict(self, subdomain: Set[Any]) -> 'Function':
        """
        Restriction: f|_S

        """
        return Function({k: v for k, v in self.mapping.items() if k in subdomain})

    def image(self, subset: Set[Any]) -> FrozenSet[Any]:
        """
        Image of a subset: f(S) = {f(x) : x ∈ S ∩ dom(f)}

        """
        return frozenset(self.mapping[x] for x in subset if x in self.mapping)

    def preimage(self, subset: Set[Any]) -> FrozenSet[Any]:
        """
        Preimage of a subset: f⁻¹(T) = {x : f(x) ∈ T}

        """
        return frozenset(x for x, y in self.mapping.items() if y in subset)

    def fixed_points(self) -> FrozenSet[Any]:
        """
        Get all fixed points: {x : f(x) = x}

        """
        return frozenset(x for x, y in self.mapping.items() if x == y)

    def orbit(self, x: Any, max_iter: int = 100) -> list:
        """
        Compute orbit of x: [x, f(x), f(f(x)), ...]

        Stops when cycle detected or max iterations reached.
        """
        result = [x]
        current = x
        seen = {x}

        for _ in range(max_iter):
            if current not in self.mapping:
                break
            next_val = self.mapping[current]
            if next_val in seen:
                break
            result.append(next_val)
            seen.add(next_val)
            current = next_val

        return result

    def cycle_structure(self) -> Dict[int, int]:
        """
        For permutations: count cycles by length.

        Returns {length: count} dict.
        """
        if not self.is_permutation():
            return {}

        seen = set()
        cycles = {}

        for x in self.mapping.keys():
            if x in seen:
                continue
            # Find cycle containing x
            cycle = []
            current = x
            while current not in seen:
                seen.add(current)
                cycle.append(current)
                current = self.mapping[current]

            length = len(cycle)
            cycles[length] = cycles.get(length, 0) + 1

        return cycles

    def __repr__(self) -> str:
        if not self.mapping:
            return "∅ → ∅"
        items = sorted(self.mapping.items(), key=lambda x: str(x[0]))
        return "{" + ", ".join(f"{k}↦{v}" for k, v in items) + "}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Function):
            return False
        return self.mapping == other.mapping

    def __hash__(self) -> int:
        return hash(frozenset(self.mapping.items()))


# =============================================================================
# Embedding Layout for Functions
# =============================================================================

# Absolute positions in 128-dim embedding:
# dims 0-7:   Domain tag
# dims 8-71:  Mapping representation (8 inputs × 8 possible outputs = 64 bits)
#             Bit (i*8 + j) = 1 iff f(i) = j
# dims 72-79: Function type one-hot [GENERAL, INJ, SUR, BIJ, CONST, ID, PERM]
# dim 80:     Domain size
# dim 81:     Range size
# dims 82-87: Property flags [injective, surjective, bijective, constant, identity, involution]

MAPPING_START = 8
MAPPING_SIZE = 64  # 8x8 matrix
TYPE_START = 72
DOMAIN_SIZE_POS = 80
RANGE_SIZE_POS = 81
PROPERTY_FLAGS_START = 82


# =============================================================================
# Function Encoder
# =============================================================================

class FunctionEncoder:
    """
    Encoder for mathematical functions.

    Supports functions with domain and codomain of size up to 8.
    All operations are deterministic.
    """

    domain_tag = SET_FUNCTION_TAG
    domain_name = "set_function"

    def __init__(self, max_size: int = 8):
        """
        Initialize encoder.

        Args:
            max_size: Maximum domain/codomain size (up to 8)
        """
        self.max_size = min(max_size, 8)
        self._input_map: Dict[Any, int] = {}
        self._output_map: Dict[Any, int] = {}
        self._reverse_input: Dict[int, Any] = {}
        self._reverse_output: Dict[int, Any] = {}

    def _input_to_index(self, element: Any) -> int:
        """Map input element to index 0-7."""
        if element in self._input_map:
            return self._input_map[element]

        if isinstance(element, int) and 0 <= element < 8:
            self._input_map[element] = element
            self._reverse_input[element] = element
            return element

        idx = len(self._input_map)
        if idx < 8:
            self._input_map[element] = idx
            self._reverse_input[idx] = element
            return idx

        return hash(element) % 8

    def _output_to_index(self, element: Any) -> int:
        """Map output element to index 0-7."""
        if element in self._output_map:
            return self._output_map[element]

        if isinstance(element, int) and 0 <= element < 8:
            self._output_map[element] = element
            self._reverse_output[element] = element
            return element

        idx = len(self._output_map)
        if idx < 8:
            self._output_map[element] = idx
            self._reverse_output[idx] = element
            return idx

        return hash(element) % 8

    def encode(self, func: Union[Function, Dict[Any, Any]]) -> Any:
        """
        Encode a function.

        Args:
            func: A Function or dict mapping

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(func, dict):
            func = Function(func)

        emb = create_embedding()

        # Domain tag (dims 0-7)
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Mapping (dims 8-71, one-hot per input)
        for inp, out in func.mapping.items():
            inp_idx = self._input_to_index(inp)
            out_idx = self._output_to_index(out)
            bit_idx = inp_idx * 8 + out_idx
            if 0 <= bit_idx < MAPPING_SIZE:
                emb = backend.at_add(emb, MAPPING_START + bit_idx, 1.0)

        # Determine function type (dims 72-79)
        if func.is_identity():
            type_idx = 5  # IDENTITY
        elif func.is_permutation():
            type_idx = 6  # PERMUTATION
        elif func.is_bijective():
            type_idx = 3  # BIJECTIVE
        elif func.is_constant():
            type_idx = 4  # CONSTANT
        elif func.is_injective():
            type_idx = 1  # INJECTIVE
        elif func.is_surjective():
            type_idx = 2  # SURJECTIVE
        else:
            type_idx = 0  # GENERAL

        emb = backend.at_add(emb, TYPE_START + type_idx, 1.0)

        # Domain and range sizes
        emb = backend.at_add(emb, DOMAIN_SIZE_POS, float(len(func.domain())))
        emb = backend.at_add(emb, RANGE_SIZE_POS, float(len(func.range())))

        # Property flags
        props = [
            func.is_injective(),
            func.is_surjective(),
            func.is_bijective(),
            func.is_constant(),
            func.is_identity(),
            func.is_involution(),
        ]
        for i, prop in enumerate(props):
            if prop:
                emb = backend.at_add(emb, PROPERTY_FLAGS_START + i, 1.0)

        return emb

    def decode(self, emb: Any) -> Function:
        """Decode embedding to a Function."""
        mapping = {}

        for inp_idx in range(8):
            for out_idx in range(8):
                bit_idx = inp_idx * 8 + out_idx
                if emb[MAPPING_START + bit_idx].item() > 0.5:
                    inp = self._reverse_input.get(inp_idx, inp_idx)
                    out = self._reverse_output.get(out_idx, out_idx)
                    mapping[inp] = out
                    break  # Function maps each input to exactly one output

        return Function(mapping)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid function."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Operations
    # =========================================================================

    def compose(self, emb1: Any, emb2: Any) -> Any:
        """
        Function composition: f₁ ∘ f₂ (apply f₂ first, then f₁)

        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # For each input i, find f₂(i) = j, then find f₁(j) = k
        for i in range(8):
            # Find what f₂ maps i to
            j = None
            for j_candidate in range(8):
                if emb2[MAPPING_START + i * 8 + j_candidate].item() > 0.5:
                    j = j_candidate
                    break

            if j is not None:
                # Find what f₁ maps j to
                for k in range(8):
                    if emb1[MAPPING_START + j * 8 + k].item() > 0.5:
                        result = backend.at_add(result, MAPPING_START + i * 8 + k, 1.0)
                        break

        return result

    def inverse(self, emb: Any) -> Optional[Any]:
        """
        Inverse function (only for bijections).

        """
        backend = get_backend()
        # Check if injective
        if emb[PROPERTY_FLAGS_START + 0].item() <= 0.5:
            return None

        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Transpose the mapping
        for i in range(8):
            for j in range(8):
                if emb[MAPPING_START + i * 8 + j].item() > 0.5:
                    result = backend.at_add(result, MAPPING_START + j * 8 + i, 1.0)

        return result

    # =========================================================================
    # Property checks
    # =========================================================================

    def is_injective(self, emb: Any) -> bool:
        """Check if function is injective from embedding."""
        return emb[PROPERTY_FLAGS_START + 0].item() > 0.5

    def is_surjective(self, emb: Any) -> bool:
        """Check if function is surjective from embedding."""
        return emb[PROPERTY_FLAGS_START + 1].item() > 0.5

    def is_bijective(self, emb: Any) -> bool:
        """Check if function is bijective from embedding."""
        return emb[PROPERTY_FLAGS_START + 2].item() > 0.5

    def apply(self, emb: Any, input_idx: int) -> Optional[int]:
        """
        Apply function to input.

        Returns output index or None if not in domain.
        """
        if not 0 <= input_idx < 8:
            return None

        for out_idx in range(8):
            if emb[MAPPING_START + input_idx * 8 + out_idx].item() > 0.5:
                return out_idx

        return None

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def encode_identity(self, n: int) -> Any:
        """Encode identity function."""
        return self.encode(Function.identity(n))

    def encode_constant(self, n: int, value: int) -> Any:
        """Encode constant function on {0,...,n-1}."""
        return self.encode(Function.constant(set(range(n)), value))

    def encode_successor(self, n: int, modular: bool = True) -> Any:
        """Encode successor function."""
        return self.encode(Function.successor(n, modular))
