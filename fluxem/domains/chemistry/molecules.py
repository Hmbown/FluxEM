"""
Molecule Encoder for Chemistry.

Embeds molecular formulas as composition multisets (element counts).
This enables stoichiometric operations:
- Combining molecules: vector addition of element counts
- Scalar multiplication: n copies of a molecule
- Mass calculation: linear combination of atomic masses

All stoichiometric operations are deterministic given integer counts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import re
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
    log_encode_value,
)
from .elements import PERIODIC_TABLE, Element


# =============================================================================
# Formula Data Structure
# =============================================================================

@dataclass
class Formula:
    """
    Molecular formula as a multiset of elements.

    Examples:
        Formula({'H': 2, 'O': 1})  # H2O
        Formula({'C': 6, 'H': 12, 'O': 6})  # C6H12O6 (glucose)
    """
    composition: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def parse(cls, formula_str: str) -> 'Formula':
        """
        Parse a molecular formula string.

        Handles: H2O, C6H12O6, NaCl, Ca(OH)2, etc.
        """
        # Tokenize: element symbols, integers, and parentheses
        tokens = re.findall(r"[A-Z][a-z]?|\d+|\(|\)", formula_str)

        # Stack of compositions for nested parentheses
        stack: List[Dict[str, int]] = [dict()]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "(":
                stack.append(dict())
                i += 1
                continue

            if tok == ")":
                if len(stack) <= 1:
                    raise ValueError(f"Unmatched ')' in formula: {formula_str}")
                group = stack.pop()
                i += 1

                mult = 1
                if i < len(tokens) and tokens[i].isdigit():
                    mult = int(tokens[i])
                    i += 1

                for symbol, count in group.items():
                    stack[-1][symbol] = stack[-1].get(symbol, 0) + count * mult
                continue

            if tok.isdigit():
                # Stray multiplier without preceding element/group
                i += 1
                continue

            # Element symbol
            symbol = tok
            i += 1
            count = 1
            if i < len(tokens) and tokens[i].isdigit():
                count = int(tokens[i])
                i += 1

            if symbol in PERIODIC_TABLE:
                stack[-1][symbol] = stack[-1].get(symbol, 0) + count

        if len(stack) != 1:
            raise ValueError(f"Unmatched '(' in formula: {formula_str}")

        return cls(stack[0])

    def molecular_weight(self) -> float:
        """Calculate molecular weight from composition."""
        mw = 0.0
        for symbol, count in self.composition.items():
            elem = PERIODIC_TABLE.get(symbol)
            if elem:
                mw += elem.atomic_mass * count
        return mw

    def atom_count(self) -> int:
        """Total number of atoms."""
        return sum(self.composition.values())

    def element_count(self) -> int:
        """Number of distinct elements."""
        return len(self.composition)

    def __add__(self, other: 'Formula') -> 'Formula':
        """Combine formulas: add element counts."""
        result = dict(self.composition)
        for symbol, count in other.composition.items():
            result[symbol] = result.get(symbol, 0) + count
        return Formula(result)

    def __mul__(self, n: int) -> 'Formula':
        """Multiply formula by scalar."""
        return Formula({s: c * n for s, c in self.composition.items()})

    def __rmul__(self, n: int) -> 'Formula':
        return self.__mul__(n)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Formula):
            return False
        return self.composition == other.composition

    def __repr__(self) -> str:
        parts = []
        # Hill system ordering: C first, H second, then remaining elements alphabetically.
        # This is the standard canonical form in chemistry (e.g., NaCl -> ClNa, H2SO4 -> H2O4S).
        # For organic compounds (containing C), this produces familiar forms like CH4, C6H12O6.
        order = ['C', 'H'] + sorted(set(self.composition.keys()) - {'C', 'H'})
        for symbol in order:
            count = self.composition.get(symbol, 0)
            if count > 0:
                if count == 1:
                    parts.append(symbol)
                else:
                    parts.append(f"{symbol}{count}")
        return ''.join(parts) if parts else "(empty)"


# Embedding layout within domain-specific region (64 dims):
# dims 0-7:   Summary [atom_count_log, element_count, mw_log, charge, ...]
# dims 8-39:  Element counts for 16 common elements (2 dims each)
# dims 40-55: Property hints (polarity, organic, etc.)
# dims 56-63: Hash fingerprint for rare elements

SUMMARY_OFFSET = 0
ELEMENT_COUNTS_OFFSET = 8
PROPERTY_HINTS_OFFSET = 40
HASH_OFFSET = 56

# Common elements in order (covers >99% of chemistry)
COMMON_ELEMENTS = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
                   'B', 'Si', 'Na', 'K', 'Ca', 'Fe']


# =============================================================================
# Molecule Encoder
# =============================================================================

class MoleculeEncoder:
    """
    Encoder for molecular formulas.

    Embeds molecules as composition vectors where:
    - Element counts are stored directly (for stoichiometric arithmetic)
    - Molecular weight is derived from composition
    - Combining molecules = adding embedding vectors

    All stoichiometric operations are deterministic given integer counts.
    """

    domain_tag = DOMAIN_TAGS["chem_molecule"]
    domain_name = "chem_molecule"

    def encode(self, formula: Union[str, Dict[str, int], Formula]) -> Any:
        """
        Encode a molecular formula.

        Args:
            formula: Formula string, dict, or Formula object

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(formula, str):
            formula = Formula.parse(formula)
        elif isinstance(formula, dict):
            formula = Formula(formula)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Summary stats
        atom_count = formula.atom_count()
        elem_count = formula.element_count()
        mw = formula.molecular_weight()

        emb = backend.at_add(emb, 8 + SUMMARY_OFFSET, backend.log(backend.array(float(atom_count + 1))))
        emb = backend.at_add(emb, 8 + SUMMARY_OFFSET + 1, float(elem_count))
        emb = backend.at_add(emb, 8 + SUMMARY_OFFSET + 2, backend.log(backend.array(mw + 1)))

        # Element counts (critical for stoichiometry)
        for i, symbol in enumerate(COMMON_ELEMENTS):
            count = formula.composition.get(symbol, 0)
            offset = ELEMENT_COUNTS_OFFSET + i * 2

            # Store: presence flag and log(count+1)
            if count > 0:
                emb = backend.at_add(emb, 8 + offset, 1.0)
                emb = backend.at_add(emb, 8 + offset + 1, backend.log(backend.array(float(count + 1))))
            # (zeros already set for absent elements)

        # Property hints
        emb = self._set_property_hints(emb, formula)

        return emb

    def decode(self, emb: Any) -> Formula:
        """
        Decode embedding to molecular formula.

        Note: The decoded formula uses canonical (Hill system) ordering:
        C first, then H, then remaining elements alphabetically.
        This means encode('NaCl').decode() produces 'ClNa' (alphabetically sorted).
        This is intentional - the embedding preserves chemical composition,
        not the original string representation. Hill ordering is the standard
        canonical form in chemistry for molecular formulas.

        Args:
            emb: 128-dim embedding

        Returns:
            Formula object with composition in canonical order
        """
        backend = get_backend()
        composition = {}

        for i, symbol in enumerate(COMMON_ELEMENTS):
            offset = ELEMENT_COUNTS_OFFSET + i * 2
            present = emb[8 + offset].item() > 0.5

            if present:
                log_count = emb[8 + offset + 1].item()
                count = int(round(backend.exp(backend.array(log_count)).item() - 1))
                if count > 0:
                    composition[symbol] = count

        return Formula(composition)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid molecule."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Stoichiometric operations
    # =========================================================================

    def combine(self, emb1: Any, emb2: Any) -> Any:
        """
        Combine two molecules (add compositions).
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Add element counts
        for i in range(len(COMMON_ELEMENTS)):
            offset = ELEMENT_COUNTS_OFFSET + i * 2

            # Get counts from both embeddings
            present1 = emb1[8 + offset].item() > 0.5
            present2 = emb2[8 + offset].item() > 0.5

            count1 = 0
            count2 = 0
            if present1:
                count1 = int(round(backend.exp(emb1[8 + offset + 1]).item() - 1))
            if present2:
                count2 = int(round(backend.exp(emb2[8 + offset + 1]).item() - 1))

            total = count1 + count2
            if total > 0:
                result = backend.at_add(result, 8 + offset, 1.0)
                result = backend.at_add(result, 8 + offset + 1, backend.log(backend.array(float(total + 1))))

        # Recompute summary
        result = self._recompute_summary(result)

        return result

    def scale(self, emb: Any, n: int) -> Any:
        """
        Multiply molecule by scalar (n copies).
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Scale element counts
        for i in range(len(COMMON_ELEMENTS)):
            offset = ELEMENT_COUNTS_OFFSET + i * 2

            present = emb[8 + offset].item() > 0.5
            if present:
                count = int(round(backend.exp(emb[8 + offset + 1]).item() - 1))
                scaled = count * n
                if scaled > 0:
                    result = backend.at_add(result, 8 + offset, 1.0)
                    result = backend.at_add(result, 8 + offset + 1, backend.log(backend.array(float(scaled + 1))))

        # Recompute summary
        result = self._recompute_summary(result)

        return result

    def subtract(self, emb1: Any, emb2: Any) -> Optional[Any]:
        """
        Subtract molecules (for reaction balancing).

        Returns None if result would have negative counts.
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Subtract element counts
        for i in range(len(COMMON_ELEMENTS)):
            offset = ELEMENT_COUNTS_OFFSET + i * 2

            present1 = emb1[8 + offset].item() > 0.5
            present2 = emb2[8 + offset].item() > 0.5

            count1 = 0
            count2 = 0
            if present1:
                count1 = int(round(backend.exp(emb1[8 + offset + 1]).item() - 1))
            if present2:
                count2 = int(round(backend.exp(emb2[8 + offset + 1]).item() - 1))

            diff = count1 - count2
            if diff < 0:
                return None  # Invalid subtraction

            if diff > 0:
                result = backend.at_add(result, 8 + offset, 1.0)
                result = backend.at_add(result, 8 + offset + 1, backend.log(backend.array(float(diff + 1))))

        # Recompute summary
        result = self._recompute_summary(result)

        return result

    def compositions_equal(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if two molecules have the same composition.
        """
        backend = get_backend()
        for i in range(len(COMMON_ELEMENTS)):
            offset = ELEMENT_COUNTS_OFFSET + i * 2

            present1 = emb1[8 + offset].item() > 0.5
            present2 = emb2[8 + offset].item() > 0.5

            if present1 != present2:
                return False

            if present1:
                count1 = round(backend.exp(emb1[8 + offset + 1]).item() - 1)
                count2 = round(backend.exp(emb2[8 + offset + 1]).item() - 1)
                if abs(count1 - count2) > 0.5:
                    return False

        return True

    def molecular_weight(self, emb: Any) -> float:
        """Calculate molecular weight from embedding."""
        formula = self.decode(emb)
        return formula.molecular_weight()

    # =========================================================================
    # Property Hints
    # =========================================================================

    def _set_property_hints(self, emb: Any, formula: Formula) -> Any:
        """Set property hint flags."""
        backend = get_backend()
        # Is organic (contains C)?
        is_organic = 'C' in formula.composition
        emb = backend.at_add(emb, 8 + PROPERTY_HINTS_OFFSET, 1.0 if is_organic else 0.0)

        # Contains halogens?
        has_halogen = any(s in formula.composition for s in ['F', 'Cl', 'Br', 'I'])
        emb = backend.at_add(emb, 8 + PROPERTY_HINTS_OFFSET + 1, 1.0 if has_halogen else 0.0)

        # Is ionic (likely)?
        has_metal = any(s in formula.composition for s in ['Na', 'K', 'Ca', 'Fe'])
        has_nonmetal = any(s in formula.composition for s in ['O', 'S', 'Cl', 'F'])
        is_ionic = has_metal and has_nonmetal
        emb = backend.at_add(emb, 8 + PROPERTY_HINTS_OFFSET + 2, 1.0 if is_ionic else 0.0)

        return emb

    def _recompute_summary(self, emb: Any) -> Any:
        """Recompute summary stats from element counts."""
        backend = get_backend()
        formula = self.decode(emb)

        atom_count = formula.atom_count()
        elem_count = formula.element_count()
        mw = formula.molecular_weight()

        emb = backend.at_add(emb, 8 + SUMMARY_OFFSET, 
            backend.log(backend.array(float(atom_count + 1))) - emb[8 + SUMMARY_OFFSET]
        )
        emb = backend.at_add(emb, 8 + SUMMARY_OFFSET + 1, float(elem_count) - emb[8 + SUMMARY_OFFSET + 1])
        emb = backend.at_add(emb, 8 + SUMMARY_OFFSET + 2, 
            backend.log(backend.array(mw + 1)) - emb[8 + SUMMARY_OFFSET + 2]
        )

        return emb

    def is_organic(self, emb: Any) -> bool:
        """Check if molecule is organic (contains carbon)."""
        return emb[8 + PROPERTY_HINTS_OFFSET].item() > 0.5

    def get_element_count(self, emb: Any, symbol: str) -> int:
        """Get count of a specific element."""
        backend = get_backend()
        if symbol not in COMMON_ELEMENTS:
            return 0

        i = COMMON_ELEMENTS.index(symbol)
        offset = ELEMENT_COUNTS_OFFSET + i * 2

        if emb[8 + offset].item() < 0.5:
            return 0

        return int(round(backend.exp(emb[8 + offset + 1]).item() - 1))
