"""
Propositional logic encoder.

Embeds propositional formulas and implements Boolean operators deterministically.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
)


# =============================================================================
# Formula Types
# =============================================================================

class FormulaType(Enum):
    """Types of propositional formulas."""
    ATOM = auto()     # Propositional variable (p, q, r, ...)
    TRUE = auto()     # Constant True
    FALSE = auto()    # Constant False
    NOT = auto()      # Negation
    AND = auto()      # Conjunction
    OR = auto()       # Disjunction
    IMPLIES = auto()  # Implication
    IFF = auto()      # Biconditional


@dataclass
class PropFormula:
    """
    Propositional formula as a tree structure.

    Parameters
    ----------
    type : FormulaType
        Type of formula.
    atom_id : int, optional
        Atom ID (for ATOM type).
    children : list[PropFormula], optional
        Child formulas.

    Examples
    --------
    >>> p = PropFormula(FormulaType.ATOM, atom_id=0)
    >>> PropFormula(FormulaType.NOT, children=[p])
    """
    type: FormulaType
    atom_id: Optional[int] = None  # For ATOM type
    children: List['PropFormula'] = field(default_factory=list)

    # Cached truth value (None if contains free variables)
    _truth_value: Optional[bool] = field(default=None, repr=False)

    @classmethod
    def atom(cls, id_or_name: Union[int, str]) -> 'PropFormula':
        """Create an atomic proposition."""
        if isinstance(id_or_name, str):
            # Convert name to id (a=0, b=1, p=15, q=16, ...)
            if len(id_or_name) == 1:
                c = id_or_name.lower()
                if 'a' <= c <= 'z':
                    id_or_name = ord(c) - ord('a')
                else:
                    id_or_name = 0
            else:
                id_or_name = hash(id_or_name) % 256
        return cls(FormulaType.ATOM, atom_id=id_or_name)

    @classmethod
    def true(cls) -> 'PropFormula':
        """Create constant True."""
        f = cls(FormulaType.TRUE)
        f._truth_value = True
        return f

    @classmethod
    def false(cls) -> 'PropFormula':
        """Create constant False."""
        f = cls(FormulaType.FALSE)
        f._truth_value = False
        return f

    def __invert__(self) -> 'PropFormula':
        """NOT operator: ~p"""
        return PropFormula(FormulaType.NOT, children=[self])

    def __and__(self, other: 'PropFormula') -> 'PropFormula':
        """AND operator: p & q"""
        return PropFormula(FormulaType.AND, children=[self, other])

    def __or__(self, other: 'PropFormula') -> 'PropFormula':
        """OR operator: p | q"""
        return PropFormula(FormulaType.OR, children=[self, other])

    def implies(self, other: 'PropFormula') -> 'PropFormula':
        """Implication: p → q"""
        return PropFormula(FormulaType.IMPLIES, children=[self, other])

    def iff(self, other: 'PropFormula') -> 'PropFormula':
        """Biconditional: p ↔ q"""
        return PropFormula(FormulaType.IFF, children=[self, other])

    def evaluate(self, assignment: Dict[int, bool]) -> Optional[bool]:
        """Evaluate formula under a truth assignment.

        Parameters
        ----------
        assignment : dict[int, bool]
            Mapping from atom id to truth value.

        Returns
        -------
        bool or None
            Truth value, or None if the assignment is incomplete.
        """
        if self.type == FormulaType.TRUE:
            return True
        elif self.type == FormulaType.FALSE:
            return False
        elif self.type == FormulaType.ATOM:
            return assignment.get(self.atom_id)
        elif self.type == FormulaType.NOT:
            child_val = self.children[0].evaluate(assignment)
            return None if child_val is None else not child_val
        elif self.type == FormulaType.AND:
            vals = [c.evaluate(assignment) for c in self.children]
            if False in vals:
                return False
            if None in vals:
                return None
            return True
        elif self.type == FormulaType.OR:
            vals = [c.evaluate(assignment) for c in self.children]
            if True in vals:
                return True
            if None in vals:
                return None
            return False
        elif self.type == FormulaType.IMPLIES:
            p = self.children[0].evaluate(assignment)
            q = self.children[1].evaluate(assignment)
            if p == False:
                return True
            if q == True:
                return True
            if p == True and q == False:
                return False
            return None
        elif self.type == FormulaType.IFF:
            p = self.children[0].evaluate(assignment)
            q = self.children[1].evaluate(assignment)
            if p is None or q is None:
                return None
            return p == q
        return None

    def atoms(self) -> Set[int]:
        """Get all atom IDs in the formula."""
        if self.type == FormulaType.ATOM:
            return {self.atom_id}
        result = set()
        for child in self.children:
            result.update(child.atoms())
        return result

    def complexity(self) -> int:
        """Count number of connectives."""
        if self.type in (FormulaType.ATOM, FormulaType.TRUE, FormulaType.FALSE):
            return 0
        return 1 + sum(c.complexity() for c in self.children)

    def is_tautology(self) -> bool:
        """Check if formula is a tautology (always true)."""
        atoms = list(self.atoms())
        if len(atoms) > 10:  # Too many variables to check
            return False

        # Check all possible assignments
        for i in range(2 ** len(atoms)):
            assignment = {atoms[j]: bool((i >> j) & 1) for j in range(len(atoms))}
            if self.evaluate(assignment) != True:
                return False
        return True

    def is_contradiction(self) -> bool:
        """Check if formula is a contradiction (always false)."""
        atoms = list(self.atoms())
        if len(atoms) > 10:
            return False

        for i in range(2 ** len(atoms)):
            assignment = {atoms[j]: bool((i >> j) & 1) for j in range(len(atoms))}
            if self.evaluate(assignment) != False:
                return False
        return True

    def is_satisfiable(self) -> bool:
        """Check if formula is satisfiable (can be true)."""
        atoms = list(self.atoms())
        if len(atoms) > 10:
            return True  # Assume satisfiable for complex formulas

        for i in range(2 ** len(atoms)):
            assignment = {atoms[j]: bool((i >> j) & 1) for j in range(len(atoms))}
            if self.evaluate(assignment) == True:
                return True
        return False

    def __repr__(self) -> str:
        if self.type == FormulaType.ATOM:
            return chr(ord('a') + (self.atom_id % 26))
        elif self.type == FormulaType.TRUE:
            return "⊤"
        elif self.type == FormulaType.FALSE:
            return "⊥"
        elif self.type == FormulaType.NOT:
            return f"¬{self.children[0]}"
        elif self.type == FormulaType.AND:
            return f"({self.children[0]} ∧ {self.children[1]})"
        elif self.type == FormulaType.OR:
            return f"({self.children[0]} ∨ {self.children[1]})"
        elif self.type == FormulaType.IMPLIES:
            return f"({self.children[0]} → {self.children[1]})"
        elif self.type == FormulaType.IFF:
            return f"({self.children[0]} ↔ {self.children[1]})"
        return "?"


# Embedding layout:
# dims 0-5:   Formula type one-hot [ATOM, TRUE/FALSE, NOT, AND, OR, IMPLIES/IFF]
# dim 6:      Truth value (1=T, 0=F, 0.5=unknown)
# dim 7:      Complexity (log scale)
# dims 8-13:  Subformula counts by type
# dim 14:     Is tautology flag
# dim 15:     Is satisfiable flag
# dims 16-31: Atom presence bitmap (for atoms 0-15)
# dims 32-47: Reserved for semantic features
# dims 48-63: Reserved

TYPE_OFFSET = 0
TRUTH_OFFSET = 6
COMPLEXITY_OFFSET = 7
SUBFORMULA_OFFSET = 8
TAUTOLOGY_FLAG = 14
SATISFIABLE_FLAG = 15
ATOM_BITMAP_OFFSET = 16


# =============================================================================
# Propositional Encoder
# =============================================================================

class PropositionalEncoder:
    """
    Encoder for propositional logic formulas.

    Truth operations are implemented deterministically.
    """

    domain_tag = DOMAIN_TAGS["logic_prop"]
    domain_name = "logic_prop"

    def encode(self, formula: PropFormula) -> Any:
        """Encode a propositional formula.

        Parameters
        ----------
        formula : PropFormula
            Input formula.

        Returns
        -------
        Any
            Embedding of shape ``(EMBEDDING_DIM,)``.
        """
        backend = get_backend()
        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Formula type (one-hot in dims 8-13)
        type_map = {
            FormulaType.ATOM: 0,
            FormulaType.TRUE: 1,
            FormulaType.FALSE: 1,
            FormulaType.NOT: 2,
            FormulaType.AND: 3,
            FormulaType.OR: 4,
            FormulaType.IMPLIES: 5,
            FormulaType.IFF: 5,
        }
        type_idx = type_map.get(formula.type, 0)
        emb = backend.at_add(emb, 8 + TYPE_OFFSET + type_idx, 1.0)

        # Truth value
        if formula.type == FormulaType.TRUE:
            emb = backend.at_add(emb, 8 + TRUTH_OFFSET, 1.0)
        elif formula.type == FormulaType.FALSE:
            emb = backend.at_add(emb, 8 + TRUTH_OFFSET, 0.0)
        else:
            # Check if it's a constant (tautology/contradiction)
            if formula.is_tautology():
                emb = backend.at_add(emb, 8 + TRUTH_OFFSET, 1.0)
            elif formula.is_contradiction():
                emb = backend.at_add(emb, 8 + TRUTH_OFFSET, 0.0)
            else:
                emb = backend.at_add(emb, 8 + TRUTH_OFFSET, 0.5)  # Unknown/variable

        # Complexity
        complexity = formula.complexity()
        emb = backend.at_add(emb, 8 + COMPLEXITY_OFFSET, backend.log(backend.array(float(complexity + 1))))

        # Subformula counts
        counts = self._count_subformulas(formula)
        for i, ftype in enumerate([FormulaType.ATOM, FormulaType.NOT,
                                   FormulaType.AND, FormulaType.OR,
                                   FormulaType.IMPLIES, FormulaType.IFF]):
            emb = backend.at_add(emb, 8 + SUBFORMULA_OFFSET + i, 
                backend.log(backend.array(float(counts.get(ftype, 0) + 1)))
            )

        # Semantic flags
        emb = backend.at_add(emb, 8 + TAUTOLOGY_FLAG, 1.0 if formula.is_tautology() else 0.0)
        emb = backend.at_add(emb, 8 + SATISFIABLE_FLAG, 1.0 if formula.is_satisfiable() else 0.0)

        # Atom bitmap
        for atom_id in formula.atoms():
            if 0 <= atom_id < 16:
                emb = backend.at_add(emb, 8 + ATOM_BITMAP_OFFSET + atom_id, 1.0)

        return emb

    def _count_subformulas(self, formula: PropFormula) -> Dict[FormulaType, int]:
        """Count subformulas by type."""
        counts = {formula.type: 1}
        for child in formula.children:
            child_counts = self._count_subformulas(child)
            for ftype, count in child_counts.items():
                counts[ftype] = counts.get(ftype, 0) + count
        return counts

    def decode(self, emb: Any) -> PropFormula:
        """Decode an embedding to a formula.

        Notes
        -----
        Full formula structure is not preserved. The decoder returns a formula
        consistent with the encoded type/truth indicators.
        """
        # Determine type from one-hot encoding
        type_idx = 0
        max_val = 0.0
        for i in range(6):
            val = emb[8 + TYPE_OFFSET + i].item()
            if val > max_val:
                max_val = val
                type_idx = i

        # Get truth value
        truth = emb[8 + TRUTH_OFFSET].item()

        if type_idx == 1:  # TRUE or FALSE
            if truth > 0.5:
                return PropFormula.true()
            else:
                return PropFormula.false()
        elif type_idx == 0:  # ATOM
            # Find first atom in bitmap
            for i in range(16):
                if emb[8 + ATOM_BITMAP_OFFSET + i].item() > 0.5:
                    return PropFormula.atom(i)
            return PropFormula.atom(0)
        else:
            # Can't fully reconstruct complex formulas
            # Return a placeholder based on truth value
            if truth > 0.9:
                return PropFormula.true()
            elif truth < 0.1:
                return PropFormula.false()
            else:
                return PropFormula.atom(0)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid propositional formula."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Lattice operations
    # =========================================================================

    def meet(self, emb1: Any, emb2: Any) -> Any:
        """
        Lattice meet operation (AND).

        Truth: min(t1, t2)
        Exact with respect to the stored truth value.
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # AND type
        result = backend.at_add(result, 8 + TYPE_OFFSET + 3, 1.0)

        # Truth value: min (AND semantics)
        t1 = emb1[8 + TRUTH_OFFSET].item()
        t2 = emb2[8 + TRUTH_OFFSET].item()
        result = backend.at_add(result, 8 + TRUTH_OFFSET, min(t1, t2))

        # Complexity adds
        c1 = emb1[8 + COMPLEXITY_OFFSET].item()
        c2 = emb2[8 + COMPLEXITY_OFFSET].item()
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, 
            backend.log(backend.exp(backend.array(c1)) + backend.exp(backend.array(c2)) + 1)
        )

        # Combine atom bitmaps (union)
        for i in range(16):
            if (emb1[8 + ATOM_BITMAP_OFFSET + i].item() > 0.5 or
                emb2[8 + ATOM_BITMAP_OFFSET + i].item() > 0.5):
                result = backend.at_add(result, 8 + ATOM_BITMAP_OFFSET + i, 1.0)

        # Update semantic flags
        # AND is tautology only if both are tautologies
        is_taut = (emb1[8 + TAUTOLOGY_FLAG].item() > 0.5 and
                   emb2[8 + TAUTOLOGY_FLAG].item() > 0.5)
        result = backend.at_add(result, 8 + TAUTOLOGY_FLAG, 1.0 if is_taut else 0.0)

        # AND is satisfiable if both are satisfiable (necessary but not sufficient)
        is_sat = (emb1[8 + SATISFIABLE_FLAG].item() > 0.5 and
                  emb2[8 + SATISFIABLE_FLAG].item() > 0.5)
        result = backend.at_add(result, 8 + SATISFIABLE_FLAG, 1.0 if is_sat else 0.0)

        return result

    def join(self, emb1: Any, emb2: Any) -> Any:
        """
        Lattice join operation (OR).

        Truth: max(t1, t2)
        Exact with respect to the stored truth value.
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # OR type
        result = backend.at_add(result, 8 + TYPE_OFFSET + 4, 1.0)

        # Truth value: max (OR semantics)
        t1 = emb1[8 + TRUTH_OFFSET].item()
        t2 = emb2[8 + TRUTH_OFFSET].item()
        result = backend.at_add(result, 8 + TRUTH_OFFSET, max(t1, t2))

        # Complexity adds
        c1 = emb1[8 + COMPLEXITY_OFFSET].item()
        c2 = emb2[8 + COMPLEXITY_OFFSET].item()
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, 
            backend.log(backend.exp(backend.array(c1)) + backend.exp(backend.array(c2)) + 1)
        )

        # Combine atom bitmaps (union)
        for i in range(16):
            if (emb1[8 + ATOM_BITMAP_OFFSET + i].item() > 0.5 or
                emb2[8 + ATOM_BITMAP_OFFSET + i].item() > 0.5):
                result = backend.at_add(result, 8 + ATOM_BITMAP_OFFSET + i, 1.0)

        # Update semantic flags
        # OR is tautology if either is tautology
        is_taut = (emb1[8 + TAUTOLOGY_FLAG].item() > 0.5 or
                   emb2[8 + TAUTOLOGY_FLAG].item() > 0.5)
        result = backend.at_add(result, 8 + TAUTOLOGY_FLAG, 1.0 if is_taut else 0.0)

        # OR is satisfiable if either is satisfiable
        is_sat = (emb1[8 + SATISFIABLE_FLAG].item() > 0.5 or
                  emb2[8 + SATISFIABLE_FLAG].item() > 0.5)
        result = backend.at_add(result, 8 + SATISFIABLE_FLAG, 1.0 if is_sat else 0.0)

        return result

    def complement(self, emb: Any) -> Any:
        """
        Complement operation (NOT).

        Truth: 1 - t
        Exact with respect to the stored truth value.
        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # NOT type
        result = backend.at_add(result, 8 + TYPE_OFFSET + 2, 1.0)

        # Truth value: complement
        t = emb[8 + TRUTH_OFFSET].item()
        result = backend.at_add(result, 8 + TRUTH_OFFSET, 1.0 - t)

        # Complexity increases by 1
        c = emb[8 + COMPLEXITY_OFFSET].item()
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, 
            backend.log(backend.exp(backend.array(c)) + 1)
        )

        # Copy atom bitmap
        for i in range(16):
            if emb[8 + ATOM_BITMAP_OFFSET + i].item() > 0.5:
                result = backend.at_add(result, 8 + ATOM_BITMAP_OFFSET + i, 1.0)

        # Swap tautology/contradiction status
        was_taut = emb[8 + TAUTOLOGY_FLAG].item() > 0.5
        was_sat = emb[8 + SATISFIABLE_FLAG].item() > 0.5

        # NOT tautology = contradiction, NOT contradiction = tautology
        if was_taut:
            result = backend.at_add(result, 8 + SATISFIABLE_FLAG, 0.0)  # contradiction
        elif not was_sat:
            result = backend.at_add(result, 8 + TAUTOLOGY_FLAG, 1.0)  # was contradiction -> tautology
            result = backend.at_add(result, 8 + SATISFIABLE_FLAG, 1.0)
        else:
            result = backend.at_add(result, 8 + SATISFIABLE_FLAG, 1.0)

        return result

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def encode_true(self) -> Any:
        """Encode constant True."""
        return self.encode(PropFormula.true())

    def encode_false(self) -> Any:
        """Encode constant False."""
        return self.encode(PropFormula.false())

    def encode_atom(self, id_or_name: Union[int, str]) -> Any:
        """Encode an atomic proposition."""
        return self.encode(PropFormula.atom(id_or_name))

    def is_tautology(self, emb: Any) -> bool:
        """Check if embedding represents a tautology."""
        return emb[8 + TAUTOLOGY_FLAG].item() > 0.5

    def is_contradiction(self, emb: Any) -> bool:
        """Check if embedding represents a contradiction."""
        return (emb[8 + SATISFIABLE_FLAG].item() < 0.5 and
                emb[8 + TRUTH_OFFSET].item() < 0.1)

    def is_satisfiable(self, emb: Any) -> bool:
        """Check if embedding represents a satisfiable formula."""
        return emb[8 + SATISFIABLE_FLAG].item() > 0.5

    def get_truth_value(self, emb: Any) -> Optional[bool]:
        """
        Get the truth value if known.

        Returns True, False, or None (for variable formulas).
        """
        t = emb[8 + TRUTH_OFFSET].item()
        if t > 0.9:
            return True
        elif t < 0.1:
            return False
        else:
            return None
