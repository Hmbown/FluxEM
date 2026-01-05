"""
Simple Type Theory Encoder.

Embeds types from a simple type system with:
- Base types
- Function types (A -> B)
- Product types (A x B)
- Sum types (A + B)
- Type checking operations
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
)


# =============================================================================
# Type Constructors
# =============================================================================

class TypeKind(Enum):
    """Kinds of types in simple type theory."""
    BASE = auto()       # Base type (e.g., Bool, Nat, Int)
    FUNCTION = auto()   # Function type (A -> B)
    PRODUCT = auto()    # Product type (A x B)
    SUM = auto()        # Sum type (A + B)
    UNIT = auto()       # Unit type (1)
    VOID = auto()       # Void/Empty type (0)
    LIST = auto()       # List type [A]
    VARIABLE = auto()   # Type variable (for polymorphism)


# Well-known base types with fixed IDs
BASE_TYPE_IDS = {
    "Unit": 0,
    "Bool": 1,
    "Nat": 2,
    "Int": 3,
    "Float": 4,
    "String": 5,
    "Char": 6,
    "Void": 7,
}


@dataclass
class Type:
    """
    Type in simple type theory.

    Examples:
        Type(TypeKind.BASE, base_name="Bool")           # Bool
        Type(TypeKind.FUNCTION, children=[A, B])        # A -> B
        Type(TypeKind.PRODUCT, children=[A, B])         # A x B
        Type(TypeKind.SUM, children=[A, B])             # A + B
    """
    kind: TypeKind
    base_name: Optional[str] = None  # For BASE and VARIABLE kinds
    base_id: int = 0                 # Numeric ID for base types
    children: List['Type'] = field(default_factory=list)

    @classmethod
    def base(cls, name: str) -> 'Type':
        """Create a base type."""
        base_id = BASE_TYPE_IDS.get(name, hash(name) % 256)
        return cls(TypeKind.BASE, base_name=name, base_id=base_id)

    @classmethod
    def variable(cls, name: str) -> 'Type':
        """Create a type variable."""
        return cls(TypeKind.VARIABLE, base_name=name)

    @classmethod
    def unit(cls) -> 'Type':
        """Create the unit type."""
        return cls(TypeKind.UNIT, base_name="Unit", base_id=0)

    @classmethod
    def void(cls) -> 'Type':
        """Create the void/empty type."""
        return cls(TypeKind.VOID, base_name="Void", base_id=7)

    @classmethod
    def function(cls, domain: 'Type', codomain: 'Type') -> 'Type':
        """Create a function type A -> B."""
        return cls(TypeKind.FUNCTION, children=[domain, codomain])

    @classmethod
    def product(cls, left: 'Type', right: 'Type') -> 'Type':
        """Create a product type A x B."""
        return cls(TypeKind.PRODUCT, children=[left, right])

    @classmethod
    def sum(cls, left: 'Type', right: 'Type') -> 'Type':
        """Create a sum type A + B."""
        return cls(TypeKind.SUM, children=[left, right])

    @classmethod
    def list_of(cls, element: 'Type') -> 'Type':
        """Create a list type [A]."""
        return cls(TypeKind.LIST, children=[element])

    def __rshift__(self, other: 'Type') -> 'Type':
        """Arrow operator: A >> B means A -> B."""
        return Type.function(self, other)

    def __mul__(self, other: 'Type') -> 'Type':
        """Product operator: A * B means A x B."""
        return Type.product(self, other)

    def __add__(self, other: 'Type') -> 'Type':
        """Sum operator: A + B."""
        return Type.sum(self, other)

    def is_ground(self) -> bool:
        """Check if type contains no type variables."""
        if self.kind == TypeKind.VARIABLE:
            return False
        for child in self.children:
            if not child.is_ground():
                return False
        return True

    def free_variables(self) -> Set[str]:
        """Get all type variable names."""
        if self.kind == TypeKind.VARIABLE:
            return {self.base_name}
        result = set()
        for child in self.children:
            result.update(child.free_variables())
        return result

    def substitute(self, var_name: str, replacement: 'Type') -> 'Type':
        """Substitute a type variable with a type."""
        if self.kind == TypeKind.VARIABLE:
            if self.base_name == var_name:
                return replacement
            return self
        elif self.kind == TypeKind.BASE:
            return self
        elif self.kind == TypeKind.UNIT:
            return self
        elif self.kind == TypeKind.VOID:
            return self
        else:
            new_children = [c.substitute(var_name, replacement) for c in self.children]
            return Type(self.kind, children=new_children)

    def size(self) -> int:
        """Get the structural size of the type."""
        if self.kind in (TypeKind.BASE, TypeKind.VARIABLE, TypeKind.UNIT, TypeKind.VOID):
            return 1
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        """Get the nesting depth of the type."""
        if self.kind in (TypeKind.BASE, TypeKind.VARIABLE, TypeKind.UNIT, TypeKind.VOID):
            return 0
        return 1 + max((c.depth() for c in self.children), default=0)

    def arrow_depth(self) -> int:
        """Get the maximum depth of function arrows."""
        if self.kind == TypeKind.FUNCTION:
            return 1 + max(c.arrow_depth() for c in self.children)
        elif self.kind in (TypeKind.BASE, TypeKind.VARIABLE, TypeKind.UNIT, TypeKind.VOID):
            return 0
        else:
            return max((c.arrow_depth() for c in self.children), default=0)

    def arity(self) -> int:
        """
        Get the arity of a function type.

        For A -> B -> C, arity is 2.
        For non-function types, arity is 0.
        """
        if self.kind != TypeKind.FUNCTION:
            return 0
        # Count arrows in right-associative chain
        count = 1
        current = self.children[1]
        while current.kind == TypeKind.FUNCTION:
            count += 1
            current = current.children[1]
        return count

    def domain(self) -> Optional['Type']:
        """Get the domain of a function type."""
        if self.kind == TypeKind.FUNCTION:
            return self.children[0]
        return None

    def codomain(self) -> Optional['Type']:
        """Get the codomain of a function type."""
        if self.kind == TypeKind.FUNCTION:
            return self.children[1]
        return None

    def left(self) -> Optional['Type']:
        """Get the left component of a product or sum type."""
        if self.kind in (TypeKind.PRODUCT, TypeKind.SUM):
            return self.children[0]
        return None

    def right(self) -> Optional['Type']:
        """Get the right component of a product or sum type."""
        if self.kind in (TypeKind.PRODUCT, TypeKind.SUM):
            return self.children[1]
        return None

    def element_type(self) -> Optional['Type']:
        """Get the element type of a list type."""
        if self.kind == TypeKind.LIST:
            return self.children[0]
        return None

    def equals(self, other: 'Type') -> bool:
        """Check structural equality of types."""
        if self.kind != other.kind:
            return False
        if self.kind == TypeKind.BASE:
            return self.base_name == other.base_name
        if self.kind == TypeKind.VARIABLE:
            return self.base_name == other.base_name
        if len(self.children) != len(other.children):
            return False
        return all(c1.equals(c2) for c1, c2 in zip(self.children, other.children))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Type):
            return False
        return self.equals(other)

    def __hash__(self) -> int:
        if self.kind in (TypeKind.BASE, TypeKind.VARIABLE):
            return hash((self.kind, self.base_name))
        return hash((self.kind, tuple(hash(c) for c in self.children)))

    def __repr__(self) -> str:
        if self.kind == TypeKind.BASE:
            return self.base_name
        elif self.kind == TypeKind.VARIABLE:
            return f"'{self.base_name}"
        elif self.kind == TypeKind.UNIT:
            return "1"
        elif self.kind == TypeKind.VOID:
            return "0"
        elif self.kind == TypeKind.FUNCTION:
            domain = self.children[0]
            codomain = self.children[1]
            # Parenthesize domain if it's a function
            if domain.kind == TypeKind.FUNCTION:
                return f"({domain}) -> {codomain}"
            return f"{domain} -> {codomain}"
        elif self.kind == TypeKind.PRODUCT:
            left = self.children[0]
            right = self.children[1]
            # Parenthesize if needed
            l_str = f"({left})" if left.kind in (TypeKind.FUNCTION, TypeKind.SUM) else str(left)
            r_str = f"({right})" if right.kind in (TypeKind.FUNCTION, TypeKind.SUM) else str(right)
            return f"{l_str} x {r_str}"
        elif self.kind == TypeKind.SUM:
            return f"({self.children[0]} + {self.children[1]})"
        elif self.kind == TypeKind.LIST:
            return f"[{self.children[0]}]"
        return "?"


# =============================================================================
# Embedding Layout for Type Theory
# =============================================================================

# Embedding layout (within dims 8-71):
# dims 0-7:   Type kind one-hot
# dim 8:      Type size (log scale)
# dim 9:      Type depth (normalized)
# dim 10:     Arrow depth (for function types)
# dim 11:     Arity (for function types)
# dim 12:     Number of type variables
# dim 13:     Is ground type flag
# dims 14-21: Base type bitmap (for 8 known base types)
# dims 22-29: Type variable hash (for 8 variables)
# dims 30-37: Structural signature
# dims 38-45: Domain type features (for functions)
# dims 46-53: Codomain/element type features
# dims 54-63: Reserved

KIND_OFFSET = 0
SIZE_OFFSET = 8
DEPTH_OFFSET = 9
ARROW_DEPTH_OFFSET = 10
ARITY_OFFSET = 11
VAR_COUNT_OFFSET = 12
GROUND_FLAG_OFFSET = 13
BASE_BITMAP_OFFSET = 14
VAR_HASH_OFFSET = 22
SIGNATURE_OFFSET = 30
DOMAIN_OFFSET = 38
CODOMAIN_OFFSET = 46


# =============================================================================
# Type Encoder
# =============================================================================

class TypeEncoder:
    """
    Encoder for simple type theory.

    Encodes types including base types, function types,
    product types, and sum types.
    """

    domain_tag = DOMAIN_TAGS["logic_type"]
    domain_name = "logic_type"

    def encode(self, typ: Type) -> Any:
        """
        Encode a type.

        Args:
            typ: Type object

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Type kind (one-hot)
        kind_map = {
            TypeKind.BASE: 0,
            TypeKind.FUNCTION: 1,
            TypeKind.PRODUCT: 2,
            TypeKind.SUM: 3,
            TypeKind.UNIT: 4,
            TypeKind.VOID: 5,
            TypeKind.LIST: 6,
            TypeKind.VARIABLE: 7,
        }
        kind_idx = kind_map.get(typ.kind, 0)
        emb = backend.at_add(emb, 8 + KIND_OFFSET + kind_idx, 1.0)

        # Size (log scale)
        size = typ.size()
        emb = backend.at_add(emb, 8 + SIZE_OFFSET, backend.log(backend.array(float(size + 1))))

        # Depth (normalized by 10)
        depth = typ.depth()
        emb = backend.at_add(emb, 8 + DEPTH_OFFSET, float(depth) / 10.0)

        # Arrow depth
        arrow_depth = typ.arrow_depth()
        emb = backend.at_add(emb, 8 + ARROW_DEPTH_OFFSET, float(arrow_depth) / 10.0)

        # Arity
        arity = typ.arity()
        emb = backend.at_add(emb, 8 + ARITY_OFFSET, float(arity) / 10.0)

        # Type variables
        free_vars = typ.free_variables()
        emb = backend.at_add(emb, 8 + VAR_COUNT_OFFSET, float(len(free_vars)) / 10.0)

        # Ground type flag
        emb = backend.at_add(emb, 8 + GROUND_FLAG_OFFSET, 1.0 if typ.is_ground() else 0.0)

        # Base type bitmap
        base_types = self._collect_base_types(typ)
        for base_name in base_types:
            if base_name in BASE_TYPE_IDS:
                idx = BASE_TYPE_IDS[base_name]
                if idx < 8:
                    emb = backend.at_add(emb, 8 + BASE_BITMAP_OFFSET + idx, 1.0)

        # Type variable hash
        for i, var_name in enumerate(sorted(free_vars)[:8]):
            var_hash = hash(var_name) % 256
            emb = backend.at_add(emb, 8 + VAR_HASH_OFFSET + i, float(var_hash) / 256.0)

        # Structural signature
        sig = self._compute_signature(typ)
        for i, val in enumerate(sig[:8]):
            emb = backend.at_add(emb, 8 + SIGNATURE_OFFSET + i, val)

        # Domain/codomain features for function types
        if typ.kind == TypeKind.FUNCTION and len(typ.children) >= 2:
            domain_features = self._type_features(typ.children[0])
            for i, val in enumerate(domain_features[:8]):
                emb = backend.at_add(emb, 8 + DOMAIN_OFFSET + i, val)

            codomain_features = self._type_features(typ.children[1])
            for i, val in enumerate(codomain_features[:8]):
                emb = backend.at_add(emb, 8 + CODOMAIN_OFFSET + i, val)
        elif typ.kind == TypeKind.LIST and len(typ.children) >= 1:
            element_features = self._type_features(typ.children[0])
            for i, val in enumerate(element_features[:8]):
                emb = backend.at_add(emb, 8 + CODOMAIN_OFFSET + i, val)

        return emb

    def _collect_base_types(self, typ: Type) -> Set[str]:
        """Collect all base type names in a type."""
        if typ.kind == TypeKind.BASE:
            return {typ.base_name}
        result = set()
        for child in typ.children:
            result.update(self._collect_base_types(child))
        return result

    def _compute_signature(self, typ: Type) -> List[float]:
        """Compute a structural signature for the type."""
        backend = get_backend()
        sig = [0.0] * 8

        # Count of each kind
        counts = self._count_kinds(typ)
        for kind, count in counts.items():
            kind_map = {
                TypeKind.BASE: 0,
                TypeKind.FUNCTION: 1,
                TypeKind.PRODUCT: 2,
                TypeKind.SUM: 3,
                TypeKind.UNIT: 4,
                TypeKind.VOID: 5,
                TypeKind.LIST: 6,
                TypeKind.VARIABLE: 7,
            }
            idx = kind_map.get(kind, 0)
            if idx < 8:
                sig[idx] = backend.log(backend.array(float(count + 1))).item()

        return sig

    def _count_kinds(self, typ: Type) -> Dict[TypeKind, int]:
        """Count occurrences of each type kind."""
        counts = {typ.kind: 1}
        for child in typ.children:
            child_counts = self._count_kinds(child)
            for kind, count in child_counts.items():
                counts[kind] = counts.get(kind, 0) + count
        return counts

    def _type_features(self, typ: Type) -> List[float]:
        """
        Extract 8-dim features for a type.

        Layout:
        - dim 0-3: Kind (BASE=0, FUNCTION=1, PRODUCT=2, SUM=3)
        - dim 4-7: For BASE types, encodes the base type ID
        """
        features = [0.0] * 8

        # Kind in first 4 dims
        kind_map = {
            TypeKind.BASE: 0,
            TypeKind.FUNCTION: 1,
            TypeKind.PRODUCT: 2,
            TypeKind.SUM: 3,
            TypeKind.UNIT: 0,  # Unit is a base type
            TypeKind.VOID: 0,  # Void is a base type
            TypeKind.LIST: 1,  # List is like a function (type constructor)
            TypeKind.VARIABLE: 0,  # Variable treated like base
        }
        idx = kind_map.get(typ.kind, 0)
        features[idx] = 1.0

        # For base types, encode the base type ID in dims 4-7
        if typ.kind == TypeKind.BASE and typ.base_name in BASE_TYPE_IDS:
            base_id = BASE_TYPE_IDS[typ.base_name]
            # Encode base_id as 4 bits
            for i in range(4):
                if (base_id >> i) & 1:
                    features[4 + i] = 1.0
        elif typ.kind == TypeKind.UNIT:
            # Unit has ID 0, no bits set
            pass
        elif typ.kind == TypeKind.VOID:
            # Void has ID 7
            features[4] = 1.0
            features[5] = 1.0
            features[6] = 1.0

        return features

    def decode(self, emb: Any) -> Type:
        """
        Decode embedding to a type.

        Note: Full structure is not preserved.
        """
        # Determine kind from one-hot encoding
        kind_idx = 0
        max_val = 0.0
        for i in range(8):
            val = emb[8 + KIND_OFFSET + i].item()
            if val > max_val:
                max_val = val
                kind_idx = i

        kind_map = {
            0: TypeKind.BASE,
            1: TypeKind.FUNCTION,
            2: TypeKind.PRODUCT,
            3: TypeKind.SUM,
            4: TypeKind.UNIT,
            5: TypeKind.VOID,
            6: TypeKind.LIST,
            7: TypeKind.VARIABLE,
        }
        kind = kind_map.get(kind_idx, TypeKind.BASE)

        if kind == TypeKind.UNIT:
            return Type.unit()
        elif kind == TypeKind.VOID:
            return Type.void()
        elif kind == TypeKind.BASE:
            # Find which base type from bitmap
            for name, idx in BASE_TYPE_IDS.items():
                if idx < 8 and emb[8 + BASE_BITMAP_OFFSET + idx].item() > 0.5:
                    return Type.base(name)
            return Type.base("Unknown")
        elif kind == TypeKind.VARIABLE:
            return Type.variable("a")
        elif kind == TypeKind.FUNCTION:
            # Can't fully reconstruct, return placeholder
            return Type.function(Type.base("A"), Type.base("B"))
        elif kind == TypeKind.PRODUCT:
            return Type.product(Type.base("A"), Type.base("B"))
        elif kind == TypeKind.SUM:
            return Type.sum(Type.base("A"), Type.base("B"))
        elif kind == TypeKind.LIST:
            return Type.list_of(Type.base("A"))
        else:
            return Type.base("Unknown")

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid type."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Type Constructor Operations
    # =========================================================================

    def encode_function(self, domain_emb: Any, codomain_emb: Any) -> Any:
        """
        Encode a function type from domain and codomain embeddings.

        Args:
            domain_emb: Embedding of the domain type
            codomain_emb: Embedding of the codomain type

        Returns:
            Embedding of domain -> codomain
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # FUNCTION kind
        result = backend.at_add(result, 8 + KIND_OFFSET + 1, 1.0)

        # Size is sum + 1
        s1 = domain_emb[8 + SIZE_OFFSET].item()
        s2 = codomain_emb[8 + SIZE_OFFSET].item()
        result = backend.at_add(result, 8 + SIZE_OFFSET, 
            backend.log(backend.exp(backend.array(s1)) + backend.exp(backend.array(s2)) + 1)
        )

        # Depth is max + 1
        d1 = domain_emb[8 + DEPTH_OFFSET].item()
        d2 = codomain_emb[8 + DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + DEPTH_OFFSET, max(d1, d2) + 0.1)

        # Arrow depth increases
        ad1 = domain_emb[8 + ARROW_DEPTH_OFFSET].item()
        ad2 = codomain_emb[8 + ARROW_DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + ARROW_DEPTH_OFFSET, max(ad1, ad2) + 0.1)

        # Arity from codomain + 1
        old_arity = codomain_emb[8 + ARITY_OFFSET].item()
        result = backend.at_add(result, 8 + ARITY_OFFSET, old_arity + 0.1)

        # Combine variable counts
        vc1 = domain_emb[8 + VAR_COUNT_OFFSET].item()
        vc2 = codomain_emb[8 + VAR_COUNT_OFFSET].item()
        result = backend.at_add(result, 8 + VAR_COUNT_OFFSET, vc1 + vc2)

        # Ground only if both ground
        g1 = domain_emb[8 + GROUND_FLAG_OFFSET].item() > 0.5
        g2 = codomain_emb[8 + GROUND_FLAG_OFFSET].item() > 0.5
        result = backend.at_add(result, 8 + GROUND_FLAG_OFFSET, 1.0 if (g1 and g2) else 0.0)

        # Combine base type bitmaps
        for i in range(8):
            v1 = domain_emb[8 + BASE_BITMAP_OFFSET + i].item()
            v2 = codomain_emb[8 + BASE_BITMAP_OFFSET + i].item()
            if v1 > 0.5 or v2 > 0.5:
                result = backend.at_add(result, 8 + BASE_BITMAP_OFFSET + i, 1.0)

        # Store domain and codomain features
        for i in range(8):
            result = backend.at_add(result, 8 + DOMAIN_OFFSET + i, 
                domain_emb[8 + KIND_OFFSET + i].item()
            )
            result = backend.at_add(result, 8 + CODOMAIN_OFFSET + i, 
                codomain_emb[8 + KIND_OFFSET + i].item()
            )

        return result

    def encode_product(self, left_emb: Any, right_emb: Any) -> Any:
        """
        Encode a product type from component embeddings.

        Args:
            left_emb: Embedding of the left type
            right_emb: Embedding of the right type

        Returns:
            Embedding of left x right
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # PRODUCT kind
        result = backend.at_add(result, 8 + KIND_OFFSET + 2, 1.0)

        # Size is sum + 1
        s1 = left_emb[8 + SIZE_OFFSET].item()
        s2 = right_emb[8 + SIZE_OFFSET].item()
        result = backend.at_add(result, 8 + SIZE_OFFSET, 
            backend.log(backend.exp(backend.array(s1)) + backend.exp(backend.array(s2)) + 1)
        )

        # Depth is max + 1
        d1 = left_emb[8 + DEPTH_OFFSET].item()
        d2 = right_emb[8 + DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + DEPTH_OFFSET, max(d1, d2) + 0.1)

        # Arrow depth is max
        ad1 = left_emb[8 + ARROW_DEPTH_OFFSET].item()
        ad2 = right_emb[8 + ARROW_DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + ARROW_DEPTH_OFFSET, max(ad1, ad2))

        # Combine variable counts
        vc1 = left_emb[8 + VAR_COUNT_OFFSET].item()
        vc2 = right_emb[8 + VAR_COUNT_OFFSET].item()
        result = backend.at_add(result, 8 + VAR_COUNT_OFFSET, vc1 + vc2)

        # Ground only if both ground
        g1 = left_emb[8 + GROUND_FLAG_OFFSET].item() > 0.5
        g2 = right_emb[8 + GROUND_FLAG_OFFSET].item() > 0.5
        result = backend.at_add(result, 8 + GROUND_FLAG_OFFSET, 1.0 if (g1 and g2) else 0.0)

        # Combine base type bitmaps
        for i in range(8):
            v1 = left_emb[8 + BASE_BITMAP_OFFSET + i].item()
            v2 = right_emb[8 + BASE_BITMAP_OFFSET + i].item()
            if v1 > 0.5 or v2 > 0.5:
                result = backend.at_add(result, 8 + BASE_BITMAP_OFFSET + i, 1.0)

        return result

    def encode_sum(self, left_emb: Any, right_emb: Any) -> Any:
        """
        Encode a sum type from component embeddings.

        Args:
            left_emb: Embedding of the left type
            right_emb: Embedding of the right type

        Returns:
            Embedding of left + right
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # SUM kind
        result = backend.at_add(result, 8 + KIND_OFFSET + 3, 1.0)

        # Size is sum + 1
        s1 = left_emb[8 + SIZE_OFFSET].item()
        s2 = right_emb[8 + SIZE_OFFSET].item()
        result = backend.at_add(result, 8 + SIZE_OFFSET, 
            backend.log(backend.exp(backend.array(s1)) + backend.exp(backend.array(s2)) + 1)
        )

        # Depth is max + 1
        d1 = left_emb[8 + DEPTH_OFFSET].item()
        d2 = right_emb[8 + DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + DEPTH_OFFSET, max(d1, d2) + 0.1)

        # Arrow depth is max
        ad1 = left_emb[8 + ARROW_DEPTH_OFFSET].item()
        ad2 = right_emb[8 + ARROW_DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + ARROW_DEPTH_OFFSET, max(ad1, ad2))

        # Combine variable counts
        vc1 = left_emb[8 + VAR_COUNT_OFFSET].item()
        vc2 = right_emb[8 + VAR_COUNT_OFFSET].item()
        result = backend.at_add(result, 8 + VAR_COUNT_OFFSET, vc1 + vc2)

        # Ground only if both ground
        g1 = left_emb[8 + GROUND_FLAG_OFFSET].item() > 0.5
        g2 = right_emb[8 + GROUND_FLAG_OFFSET].item() > 0.5
        result = backend.at_add(result, 8 + GROUND_FLAG_OFFSET, 1.0 if (g1 and g2) else 0.0)

        # Combine base type bitmaps
        for i in range(8):
            v1 = left_emb[8 + BASE_BITMAP_OFFSET + i].item()
            v2 = right_emb[8 + BASE_BITMAP_OFFSET + i].item()
            if v1 > 0.5 or v2 > 0.5:
                result = backend.at_add(result, 8 + BASE_BITMAP_OFFSET + i, 1.0)

        return result

    # =========================================================================
    # Type Checking Operations
    # =========================================================================

    def types_equal(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if two type embeddings represent equal types.

        This is approximate - full equality requires the actual types.
        """
        if not self.is_valid(emb1) or not self.is_valid(emb2):
            return False

        # Compare kind
        kind1 = -1
        kind2 = -1
        max_val1 = 0.0
        max_val2 = 0.0
        for i in range(8):
            v1 = emb1[8 + KIND_OFFSET + i].item()
            v2 = emb2[8 + KIND_OFFSET + i].item()
            if v1 > max_val1:
                max_val1 = v1
                kind1 = i
            if v2 > max_val2:
                max_val2 = v2
                kind2 = i

        if kind1 != kind2:
            return False

        # Compare size
        s1 = emb1[8 + SIZE_OFFSET].item()
        s2 = emb2[8 + SIZE_OFFSET].item()
        if abs(s1 - s2) > 0.1:
            return False

        # Compare depth
        d1 = emb1[8 + DEPTH_OFFSET].item()
        d2 = emb2[8 + DEPTH_OFFSET].item()
        if abs(d1 - d2) > 0.1:
            return False

        # Compare base type bitmap
        for i in range(8):
            b1 = emb1[8 + BASE_BITMAP_OFFSET + i].item() > 0.5
            b2 = emb2[8 + BASE_BITMAP_OFFSET + i].item() > 0.5
            if b1 != b2:
                return False

        return True

    def is_subtype(self, sub_emb: Any, super_emb: Any) -> bool:
        """
        Check if one type is a subtype of another.

        For simple type theory without subtyping, this is just equality.
        Override in extensions for structural subtyping.
        """
        return self.types_equal(sub_emb, super_emb)

    def can_apply(self, func_emb: Any, arg_emb: Any) -> bool:
        """
        Check if a function type can be applied to an argument type.

        Returns True if func is A -> B and arg matches A.
        """
        if not self.is_valid(func_emb) or not self.is_valid(arg_emb):
            return False

        # Check if func is a function type
        func_kind = 0
        max_val = 0.0
        for i in range(8):
            val = func_emb[8 + KIND_OFFSET + i].item()
            if val > max_val:
                max_val = val
                func_kind = i

        if func_kind != 1:  # Not a function
            return False

        # Get arg's type features
        arg_kind = 0
        max_val = 0.0
        for i in range(8):
            val = arg_emb[8 + KIND_OFFSET + i].item()
            if val > max_val:
                max_val = val
                arg_kind = i

        # Compare domain features with arg features
        # First check kind match (dims 0-3 in the feature encoding)
        for i in range(4):
            domain_val = func_emb[8 + DOMAIN_OFFSET + i].item()
            # For arg, we need to check its kind
            arg_val = 1.0 if i == arg_kind else 0.0
            if domain_val > 0.5 and arg_val < 0.5:
                return False
            if domain_val < 0.5 and arg_val > 0.5:
                return False

        # For base types, also check the base type ID (dims 4-7)
        if arg_kind == 0:  # arg is a base type
            # Extract arg's base type ID from its bitmap
            for i in range(4):
                domain_base_bit = func_emb[8 + DOMAIN_OFFSET + 4 + i].item()
                # Find arg's base type from its bitmap
                arg_base_bit = 0.0
                for name, idx in BASE_TYPE_IDS.items():
                    if idx < 8 and arg_emb[8 + BASE_BITMAP_OFFSET + idx].item() > 0.5:
                        # Found the base type, check bit i
                        if (idx >> i) & 1:
                            arg_base_bit = 1.0
                        break

                if domain_base_bit > 0.5 and arg_base_bit < 0.5:
                    return False
                if domain_base_bit < 0.5 and arg_base_bit > 0.5:
                    return False

        return True

    def apply_function(self, func_emb: Any, arg_emb: Any) -> Optional[Any]:
        """
        Apply a function type to an argument, returning the result type.

        If func is A -> B and arg is A, returns embedding for B.
        """
        backend = get_backend()
        if not self.can_apply(func_emb, arg_emb):
            return None

        # Return the codomain as the result
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Copy codomain features as the new type's kind
        for i in range(8):
            codomain_val = func_emb[8 + CODOMAIN_OFFSET + i].item()
            result = backend.at_add(result, 8 + KIND_OFFSET + i, codomain_val)

        # Reduce arrow depth
        ad = func_emb[8 + ARROW_DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + ARROW_DEPTH_OFFSET, max(0, ad - 0.1))

        # Reduce arity
        ar = func_emb[8 + ARITY_OFFSET].item()
        result = backend.at_add(result, 8 + ARITY_OFFSET, max(0, ar - 0.1))

        # Keep other features from func
        result = backend.at_add(result, 8 + VAR_COUNT_OFFSET, func_emb[8 + VAR_COUNT_OFFSET].item())
        result = backend.at_add(result, 8 + GROUND_FLAG_OFFSET, func_emb[8 + GROUND_FLAG_OFFSET].item())

        return result

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def encode_base(self, name: str) -> Any:
        """Encode a base type by name."""
        return self.encode(Type.base(name))

    def encode_bool(self) -> Any:
        """Encode Bool type."""
        return self.encode(Type.base("Bool"))

    def encode_nat(self) -> Any:
        """Encode Nat type."""
        return self.encode(Type.base("Nat"))

    def encode_int(self) -> Any:
        """Encode Int type."""
        return self.encode(Type.base("Int"))

    def encode_unit(self) -> Any:
        """Encode Unit type."""
        return self.encode(Type.unit())

    def encode_void(self) -> Any:
        """Encode Void type."""
        return self.encode(Type.void())

    def get_kind(self, emb: Any) -> TypeKind:
        """Get the type kind from an embedding."""
        kind_idx = 0
        max_val = 0.0
        for i in range(8):
            val = emb[8 + KIND_OFFSET + i].item()
            if val > max_val:
                max_val = val
                kind_idx = i

        kind_map = {
            0: TypeKind.BASE,
            1: TypeKind.FUNCTION,
            2: TypeKind.PRODUCT,
            3: TypeKind.SUM,
            4: TypeKind.UNIT,
            5: TypeKind.VOID,
            6: TypeKind.LIST,
            7: TypeKind.VARIABLE,
        }
        return kind_map.get(kind_idx, TypeKind.BASE)

    def is_function_type(self, emb: Any) -> bool:
        """Check if embedding represents a function type."""
        return self.get_kind(emb) == TypeKind.FUNCTION

    def is_product_type(self, emb: Any) -> bool:
        """Check if embedding represents a product type."""
        return self.get_kind(emb) == TypeKind.PRODUCT

    def is_sum_type(self, emb: Any) -> bool:
        """Check if embedding represents a sum type."""
        return self.get_kind(emb) == TypeKind.SUM

    def is_ground_type(self, emb: Any) -> bool:
        """Check if embedding represents a ground (no variables) type."""
        return emb[8 + GROUND_FLAG_OFFSET].item() > 0.5

    def get_arity(self, emb: Any) -> int:
        """Get the arity of a function type."""
        return int(emb[8 + ARITY_OFFSET].item() * 10.0 + 0.5)
