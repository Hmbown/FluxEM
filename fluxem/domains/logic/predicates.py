"""
First-Order Logic (Predicate Logic) Encoder.

Embeds first-order logic formulas with:
- Predicates with arity
- Quantifiers (universal and existential)
- Variable binding and scope tracking
- Terms (variables, constants, functions)
- Substitution and unification operations
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
# Term Types
# =============================================================================

class TermType(Enum):
    """Types of first-order terms."""
    VARIABLE = auto()    # x, y, z, ...
    CONSTANT = auto()    # a, b, c, ...
    FUNCTION = auto()    # f(t1, ..., tn)


@dataclass
class Term:
    """
    First-order term.

    Examples:
        Term(TermType.VARIABLE, name="x")       # variable x
        Term(TermType.CONSTANT, name="a")       # constant a
        Term(TermType.FUNCTION, name="f", args=[x, y])  # f(x, y)
    """
    type: TermType
    name: str
    args: List['Term'] = field(default_factory=list)

    @classmethod
    def variable(cls, name: str) -> 'Term':
        """Create a variable term."""
        return cls(TermType.VARIABLE, name)

    @classmethod
    def constant(cls, name: str) -> 'Term':
        """Create a constant term."""
        return cls(TermType.CONSTANT, name)

    @classmethod
    def function(cls, name: str, args: List['Term']) -> 'Term':
        """Create a function term."""
        return cls(TermType.FUNCTION, name, args)

    def free_variables(self) -> Set[str]:
        """Get all free variable names in the term."""
        if self.type == TermType.VARIABLE:
            return {self.name}
        elif self.type == TermType.CONSTANT:
            return set()
        else:  # FUNCTION
            result = set()
            for arg in self.args:
                result.update(arg.free_variables())
            return result

    def substitute(self, var_name: str, replacement: 'Term') -> 'Term':
        """Substitute a variable with a term."""
        if self.type == TermType.VARIABLE:
            if self.name == var_name:
                return replacement
            return self
        elif self.type == TermType.CONSTANT:
            return self
        else:  # FUNCTION
            new_args = [arg.substitute(var_name, replacement) for arg in self.args]
            return Term.function(self.name, new_args)

    def depth(self) -> int:
        """Get the depth of the term."""
        if self.type in (TermType.VARIABLE, TermType.CONSTANT):
            return 0
        return 1 + max((arg.depth() for arg in self.args), default=0)

    def __repr__(self) -> str:
        if self.type == TermType.VARIABLE:
            return self.name
        elif self.type == TermType.CONSTANT:
            return self.name
        else:
            args_str = ", ".join(repr(a) for a in self.args)
            return f"{self.name}({args_str})"


# =============================================================================
# Formula Types
# =============================================================================

class FOLFormulaType(Enum):
    """Types of first-order logic formulas."""
    PREDICATE = auto()   # P(t1, ..., tn)
    TRUE = auto()        # Constant True
    FALSE = auto()       # Constant False
    NOT = auto()         # Negation
    AND = auto()         # Conjunction
    OR = auto()          # Disjunction
    IMPLIES = auto()     # Implication
    IFF = auto()         # Biconditional
    FORALL = auto()      # Universal quantifier
    EXISTS = auto()      # Existential quantifier
    EQUALS = auto()      # Equality predicate


@dataclass
class FOLFormula:
    """
    First-order logic formula.

    Examples:
        FOLFormula(FOLFormulaType.PREDICATE, pred_name="P", terms=[x, y])  # P(x, y)
        FOLFormula(FOLFormulaType.FORALL, var="x", children=[phi])  # ∀x.φ
        FOLFormula(FOLFormulaType.EXISTS, var="x", children=[phi])  # ∃x.φ
    """
    type: FOLFormulaType
    pred_name: Optional[str] = None  # For PREDICATE type
    pred_arity: int = 0              # Arity of predicate
    terms: List[Term] = field(default_factory=list)  # For PREDICATE and EQUALS
    var: Optional[str] = None        # Bound variable for FORALL/EXISTS
    children: List['FOLFormula'] = field(default_factory=list)

    # Cached quantifier depth
    _quant_depth: int = field(default=0, repr=False)

    @classmethod
    def predicate(cls, name: str, terms: List[Term]) -> 'FOLFormula':
        """Create a predicate formula P(t1, ..., tn)."""
        return cls(FOLFormulaType.PREDICATE, pred_name=name,
                   pred_arity=len(terms), terms=terms)

    @classmethod
    def equals(cls, t1: Term, t2: Term) -> 'FOLFormula':
        """Create an equality formula t1 = t2."""
        return cls(FOLFormulaType.EQUALS, terms=[t1, t2])

    @classmethod
    def true(cls) -> 'FOLFormula':
        """Create constant True."""
        return cls(FOLFormulaType.TRUE)

    @classmethod
    def false(cls) -> 'FOLFormula':
        """Create constant False."""
        return cls(FOLFormulaType.FALSE)

    @classmethod
    def forall(cls, var: str, formula: 'FOLFormula') -> 'FOLFormula':
        """Create universal quantification ∀var.formula."""
        f = cls(FOLFormulaType.FORALL, var=var, children=[formula])
        f._quant_depth = formula._quant_depth + 1
        return f

    @classmethod
    def exists(cls, var: str, formula: 'FOLFormula') -> 'FOLFormula':
        """Create existential quantification ∃var.formula."""
        f = cls(FOLFormulaType.EXISTS, var=var, children=[formula])
        f._quant_depth = formula._quant_depth + 1
        return f

    def __invert__(self) -> 'FOLFormula':
        """NOT operator: ~phi"""
        f = FOLFormula(FOLFormulaType.NOT, children=[self])
        f._quant_depth = self._quant_depth
        return f

    def __and__(self, other: 'FOLFormula') -> 'FOLFormula':
        """AND operator: phi & psi"""
        f = FOLFormula(FOLFormulaType.AND, children=[self, other])
        f._quant_depth = max(self._quant_depth, other._quant_depth)
        return f

    def __or__(self, other: 'FOLFormula') -> 'FOLFormula':
        """OR operator: phi | psi"""
        f = FOLFormula(FOLFormulaType.OR, children=[self, other])
        f._quant_depth = max(self._quant_depth, other._quant_depth)
        return f

    def implies(self, other: 'FOLFormula') -> 'FOLFormula':
        """Implication: phi -> psi"""
        f = FOLFormula(FOLFormulaType.IMPLIES, children=[self, other])
        f._quant_depth = max(self._quant_depth, other._quant_depth)
        return f

    def iff(self, other: 'FOLFormula') -> 'FOLFormula':
        """Biconditional: phi <-> psi"""
        f = FOLFormula(FOLFormulaType.IFF, children=[self, other])
        f._quant_depth = max(self._quant_depth, other._quant_depth)
        return f

    def free_variables(self) -> Set[str]:
        """Get all free (unbound) variables in the formula."""
        if self.type == FOLFormulaType.PREDICATE:
            result = set()
            for term in self.terms:
                result.update(term.free_variables())
            return result
        elif self.type == FOLFormulaType.EQUALS:
            result = set()
            for term in self.terms:
                result.update(term.free_variables())
            return result
        elif self.type in (FOLFormulaType.TRUE, FOLFormulaType.FALSE):
            return set()
        elif self.type in (FOLFormulaType.FORALL, FOLFormulaType.EXISTS):
            # Bound variable is removed from free variables
            child_free = self.children[0].free_variables()
            return child_free - {self.var}
        else:
            result = set()
            for child in self.children:
                result.update(child.free_variables())
            return result

    def bound_variables(self) -> Set[str]:
        """Get all bound variables in the formula."""
        if self.type in (FOLFormulaType.FORALL, FOLFormulaType.EXISTS):
            child_bound = self.children[0].bound_variables()
            return child_bound | {self.var}
        else:
            result = set()
            for child in self.children:
                result.update(child.bound_variables())
            return result

    def predicates(self) -> Set[Tuple[str, int]]:
        """Get all predicates with their arities."""
        if self.type == FOLFormulaType.PREDICATE:
            return {(self.pred_name, self.pred_arity)}
        elif self.type == FOLFormulaType.EQUALS:
            return {("=", 2)}
        else:
            result = set()
            for child in self.children:
                result.update(child.predicates())
            return result

    def quantifier_depth(self) -> int:
        """Get the maximum nesting depth of quantifiers."""
        if self.type in (FOLFormulaType.FORALL, FOLFormulaType.EXISTS):
            return 1 + self.children[0].quantifier_depth()
        elif self.type in (FOLFormulaType.PREDICATE, FOLFormulaType.EQUALS,
                           FOLFormulaType.TRUE, FOLFormulaType.FALSE):
            return 0
        else:
            return max((c.quantifier_depth() for c in self.children), default=0)

    def complexity(self) -> int:
        """Count number of connectives and quantifiers."""
        if self.type in (FOLFormulaType.PREDICATE, FOLFormulaType.EQUALS,
                         FOLFormulaType.TRUE, FOLFormulaType.FALSE):
            return 0
        return 1 + sum(c.complexity() for c in self.children)

    def is_sentence(self) -> bool:
        """Check if formula is a sentence (no free variables)."""
        return len(self.free_variables()) == 0

    def substitute(self, var_name: str, term: Term) -> 'FOLFormula':
        """Substitute a free variable with a term."""
        if self.type == FOLFormulaType.PREDICATE:
            new_terms = [t.substitute(var_name, term) for t in self.terms]
            return FOLFormula.predicate(self.pred_name, new_terms)
        elif self.type == FOLFormulaType.EQUALS:
            new_terms = [t.substitute(var_name, term) for t in self.terms]
            return FOLFormula.equals(new_terms[0], new_terms[1])
        elif self.type in (FOLFormulaType.TRUE, FOLFormulaType.FALSE):
            return self
        elif self.type in (FOLFormulaType.FORALL, FOLFormulaType.EXISTS):
            if self.var == var_name:
                # Variable is bound here, don't substitute
                return self
            # Check for variable capture
            term_vars = term.free_variables()
            if self.var in term_vars:
                # Would cause variable capture - need alpha conversion
                # For simplicity, we just skip substitution here
                return self
            new_child = self.children[0].substitute(var_name, term)
            if self.type == FOLFormulaType.FORALL:
                return FOLFormula.forall(self.var, new_child)
            else:
                return FOLFormula.exists(self.var, new_child)
        else:
            new_children = [c.substitute(var_name, term) for c in self.children]
            f = FOLFormula(self.type, children=new_children)
            f._quant_depth = self._quant_depth
            return f

    def __repr__(self) -> str:
        if self.type == FOLFormulaType.PREDICATE:
            if self.terms:
                args = ", ".join(repr(t) for t in self.terms)
                return f"{self.pred_name}({args})"
            return self.pred_name
        elif self.type == FOLFormulaType.EQUALS:
            return f"({self.terms[0]} = {self.terms[1]})"
        elif self.type == FOLFormulaType.TRUE:
            return "T"
        elif self.type == FOLFormulaType.FALSE:
            return "F"
        elif self.type == FOLFormulaType.NOT:
            return f"~{self.children[0]}"
        elif self.type == FOLFormulaType.AND:
            return f"({self.children[0]} & {self.children[1]})"
        elif self.type == FOLFormulaType.OR:
            return f"({self.children[0]} | {self.children[1]})"
        elif self.type == FOLFormulaType.IMPLIES:
            return f"({self.children[0]} -> {self.children[1]})"
        elif self.type == FOLFormulaType.IFF:
            return f"({self.children[0]} <-> {self.children[1]})"
        elif self.type == FOLFormulaType.FORALL:
            return f"(A{self.var}.{self.children[0]})"
        elif self.type == FOLFormulaType.EXISTS:
            return f"(E{self.var}.{self.children[0]})"
        return "?"


# =============================================================================
# Embedding Layout for First-Order Logic
# =============================================================================

# Embedding layout (within dims 8-71):
# dims 0-9:   Formula type one-hot
# dim 10:     Quantifier depth (normalized)
# dim 11:     Complexity (log scale)
# dim 12:     Number of free variables
# dim 13:     Number of bound variables
# dim 14:     Number of distinct predicates
# dim 15:     Max predicate arity
# dims 16-23: Subformula counts by type
# dim 24:     Is sentence flag (no free vars)
# dim 25:     Is in prenex normal form
# dims 26-33: Variable hash (for 8 variables)
# dims 34-41: Predicate hash (for 8 predicates)
# dims 42-49: Term depth encoding
# dims 50-63: Reserved

TYPE_OFFSET = 0
QUANT_DEPTH_OFFSET = 10
COMPLEXITY_OFFSET = 11
FREE_VARS_OFFSET = 12
BOUND_VARS_OFFSET = 13
PRED_COUNT_OFFSET = 14
MAX_ARITY_OFFSET = 15
SUBFORMULA_OFFSET = 16
SENTENCE_FLAG = 24
PRENEX_FLAG = 25
VAR_HASH_OFFSET = 26
PRED_HASH_OFFSET = 34
TERM_DEPTH_OFFSET = 42


# =============================================================================
# Predicate Logic Encoder
# =============================================================================

class PredicateEncoder:
    """
    Encoder for first-order logic formulas.

    Encodes predicates, quantifiers, variable binding, and terms.
    """

    domain_tag = DOMAIN_TAGS["logic_pred"]
    domain_name = "logic_pred"

    def encode(self, formula: FOLFormula) -> Any:
        """
        Encode a first-order logic formula.

        Args:
            formula: FOLFormula object

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Formula type (one-hot in dims 8+TYPE_OFFSET to 8+TYPE_OFFSET+9)
        type_map = {
            FOLFormulaType.PREDICATE: 0,
            FOLFormulaType.TRUE: 1,
            FOLFormulaType.FALSE: 1,
            FOLFormulaType.NOT: 2,
            FOLFormulaType.AND: 3,
            FOLFormulaType.OR: 4,
            FOLFormulaType.IMPLIES: 5,
            FOLFormulaType.IFF: 5,
            FOLFormulaType.FORALL: 6,
            FOLFormulaType.EXISTS: 7,
            FOLFormulaType.EQUALS: 8,
        }
        type_idx = type_map.get(formula.type, 0)
        emb = backend.at_add(emb, 8 + TYPE_OFFSET + type_idx, 1.0)

        # Quantifier depth (normalized by 10)
        quant_depth = formula.quantifier_depth()
        emb = backend.at_add(emb, 8 + QUANT_DEPTH_OFFSET, float(quant_depth) / 10.0)

        # Complexity
        complexity = formula.complexity()
        emb = backend.at_add(emb, 8 + COMPLEXITY_OFFSET, backend.log(backend.array(float(complexity + 1))))

        # Variable counts
        free_vars = formula.free_variables()
        bound_vars = formula.bound_variables()
        emb = backend.at_add(emb, 8 + FREE_VARS_OFFSET, float(len(free_vars)) / 10.0)
        emb = backend.at_add(emb, 8 + BOUND_VARS_OFFSET, float(len(bound_vars)) / 10.0)

        # Predicate info
        preds = formula.predicates()
        emb = backend.at_add(emb, 8 + PRED_COUNT_OFFSET, float(len(preds)) / 10.0)
        max_arity = max((arity for _, arity in preds), default=0)
        emb = backend.at_add(emb, 8 + MAX_ARITY_OFFSET, float(max_arity) / 10.0)

        # Subformula counts
        counts = self._count_subformulas(formula)
        for i, ftype in enumerate([FOLFormulaType.PREDICATE, FOLFormulaType.NOT,
                                   FOLFormulaType.AND, FOLFormulaType.OR,
                                   FOLFormulaType.IMPLIES, FOLFormulaType.FORALL,
                                   FOLFormulaType.EXISTS, FOLFormulaType.EQUALS]):
            if i < 8:
                emb = backend.at_add(emb, 8 + SUBFORMULA_OFFSET + i, 
                    backend.log(backend.array(float(counts.get(ftype, 0) + 1)))
                )

        # Sentence flag
        emb = backend.at_add(emb, 8 + SENTENCE_FLAG, 1.0 if formula.is_sentence() else 0.0)

        # Prenex normal form flag (all quantifiers at front)
        emb = backend.at_add(emb, 8 + PRENEX_FLAG, 1.0 if self._is_prenex(formula) else 0.0)

        # Variable hash
        for i, var in enumerate(sorted(free_vars | bound_vars)[:8]):
            var_hash = hash(var) % 256
            emb = backend.at_add(emb, 8 + VAR_HASH_OFFSET + i, float(var_hash) / 256.0)

        # Predicate hash
        for i, (pred_name, arity) in enumerate(sorted(preds)[:8]):
            pred_hash = (hash(pred_name) + arity) % 256
            emb = backend.at_add(emb, 8 + PRED_HASH_OFFSET + i, float(pred_hash) / 256.0)

        # Term depth encoding
        term_depth = self._max_term_depth(formula)
        emb = backend.at_add(emb, 8 + TERM_DEPTH_OFFSET, float(term_depth) / 10.0)

        return emb

    def _count_subformulas(self, formula: FOLFormula) -> Dict[FOLFormulaType, int]:
        """Count subformulas by type."""
        counts = {formula.type: 1}
        for child in formula.children:
            child_counts = self._count_subformulas(child)
            for ftype, count in child_counts.items():
                counts[ftype] = counts.get(ftype, 0) + count
        return counts

    def _is_prenex(self, formula: FOLFormula) -> bool:
        """Check if formula is in prenex normal form."""
        # Find the matrix (quantifier-free part)
        current = formula
        while current.type in (FOLFormulaType.FORALL, FOLFormulaType.EXISTS):
            current = current.children[0]

        # Check if matrix is quantifier-free
        return self._is_quantifier_free(current)

    def _is_quantifier_free(self, formula: FOLFormula) -> bool:
        """Check if formula has no quantifiers."""
        if formula.type in (FOLFormulaType.FORALL, FOLFormulaType.EXISTS):
            return False
        for child in formula.children:
            if not self._is_quantifier_free(child):
                return False
        return True

    def _max_term_depth(self, formula: FOLFormula) -> int:
        """Get maximum depth of terms in the formula."""
        if formula.type == FOLFormulaType.PREDICATE:
            return max((t.depth() for t in formula.terms), default=0)
        elif formula.type == FOLFormulaType.EQUALS:
            return max((t.depth() for t in formula.terms), default=0)
        else:
            return max((self._max_term_depth(c) for c in formula.children), default=0)

    def decode(self, emb: Any) -> FOLFormula:
        """
        Decode embedding to a formula.

        Note: Full structure is not preserved - only basic type info.
        """
        # Determine type from one-hot encoding
        type_idx = 0
        max_val = 0.0
        for i in range(10):
            val = emb[8 + TYPE_OFFSET + i].item()
            if val > max_val:
                max_val = val
                type_idx = i

        if type_idx == 1:  # TRUE or FALSE
            truth = emb[8 + QUANT_DEPTH_OFFSET].item()
            if truth > 0.5:
                return FOLFormula.true()
            return FOLFormula.false()
        elif type_idx == 0:  # PREDICATE
            return FOLFormula.predicate("P", [])
        elif type_idx == 6:  # FORALL
            return FOLFormula.forall("x", FOLFormula.predicate("P", [Term.variable("x")]))
        elif type_idx == 7:  # EXISTS
            return FOLFormula.exists("x", FOLFormula.predicate("P", [Term.variable("x")]))
        else:
            # Default to a simple predicate
            return FOLFormula.predicate("P", [])

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid first-order formula."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Quantifier Operations
    # =========================================================================

    def encode_forall(self, var: str, body_emb: Any) -> Any:
        """
        Encode universal quantification over a formula embedding.

        Args:
            var: Variable name to bind
            body_emb: Embedding of the body formula

        Returns:
            Embedding of ∀var.body
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # FORALL type
        result = backend.at_add(result, 8 + TYPE_OFFSET + 6, 1.0)

        # Increment quantifier depth
        old_depth = body_emb[8 + QUANT_DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + QUANT_DEPTH_OFFSET, old_depth + 0.1)

        # Increment complexity
        old_complexity = body_emb[8 + COMPLEXITY_OFFSET].item()
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, old_complexity + 0.1)

        # Decrement free vars, increment bound vars
        old_free = body_emb[8 + FREE_VARS_OFFSET].item()
        old_bound = body_emb[8 + BOUND_VARS_OFFSET].item()
        result = backend.at_add(result, 8 + FREE_VARS_OFFSET, max(0, old_free - 0.1))
        result = backend.at_add(result, 8 + BOUND_VARS_OFFSET, old_bound + 0.1)

        # Copy predicate info
        result = backend.at_add(result, 8 + PRED_COUNT_OFFSET, body_emb[8 + PRED_COUNT_OFFSET].item())
        result = backend.at_add(result, 8 + MAX_ARITY_OFFSET, body_emb[8 + MAX_ARITY_OFFSET].item())

        # Update sentence flag
        if old_free <= 0.1:
            result = backend.at_add(result, 8 + SENTENCE_FLAG, 1.0)

        return result

    def encode_exists(self, var: str, body_emb: Any) -> Any:
        """
        Encode existential quantification over a formula embedding.

        Args:
            var: Variable name to bind
            body_emb: Embedding of the body formula

        Returns:
            Embedding of ∃var.body
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # EXISTS type
        result = backend.at_add(result, 8 + TYPE_OFFSET + 7, 1.0)

        # Increment quantifier depth
        old_depth = body_emb[8 + QUANT_DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + QUANT_DEPTH_OFFSET, old_depth + 0.1)

        # Increment complexity
        old_complexity = body_emb[8 + COMPLEXITY_OFFSET].item()
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, old_complexity + 0.1)

        # Decrement free vars, increment bound vars
        old_free = body_emb[8 + FREE_VARS_OFFSET].item()
        old_bound = body_emb[8 + BOUND_VARS_OFFSET].item()
        result = backend.at_add(result, 8 + FREE_VARS_OFFSET, max(0, old_free - 0.1))
        result = backend.at_add(result, 8 + BOUND_VARS_OFFSET, old_bound + 0.1)

        # Copy predicate info
        result = backend.at_add(result, 8 + PRED_COUNT_OFFSET, body_emb[8 + PRED_COUNT_OFFSET].item())
        result = backend.at_add(result, 8 + MAX_ARITY_OFFSET, body_emb[8 + MAX_ARITY_OFFSET].item())

        return result

    # =========================================================================
    # Connective Operations
    # =========================================================================

    def negate(self, emb: Any) -> Any:
        """Negate a formula embedding."""
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # NOT type
        result = backend.at_add(result, 8 + TYPE_OFFSET + 2, 1.0)

        # Keep quantifier depth
        result = backend.at_add(result, 8 + QUANT_DEPTH_OFFSET, emb[8 + QUANT_DEPTH_OFFSET].item())

        # Increment complexity
        old_complexity = emb[8 + COMPLEXITY_OFFSET].item()
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, old_complexity + 0.1)

        # Copy other features
        result = backend.at_add(result, 8 + FREE_VARS_OFFSET, emb[8 + FREE_VARS_OFFSET].item())
        result = backend.at_add(result, 8 + BOUND_VARS_OFFSET, emb[8 + BOUND_VARS_OFFSET].item())
        result = backend.at_add(result, 8 + PRED_COUNT_OFFSET, emb[8 + PRED_COUNT_OFFSET].item())
        result = backend.at_add(result, 8 + MAX_ARITY_OFFSET, emb[8 + MAX_ARITY_OFFSET].item())
        result = backend.at_add(result, 8 + SENTENCE_FLAG, emb[8 + SENTENCE_FLAG].item())

        return result

    def conjoin(self, emb1: Any, emb2: Any) -> Any:
        """Conjoin two formula embeddings (AND)."""
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # AND type
        result = backend.at_add(result, 8 + TYPE_OFFSET + 3, 1.0)

        # Max quantifier depth
        d1 = emb1[8 + QUANT_DEPTH_OFFSET].item()
        d2 = emb2[8 + QUANT_DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + QUANT_DEPTH_OFFSET, max(d1, d2))

        # Add complexities
        c1 = emb1[8 + COMPLEXITY_OFFSET].item()
        c2 = emb2[8 + COMPLEXITY_OFFSET].item()
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, c1 + c2 + 0.1)

        # Combine variables (approximate)
        fv1 = emb1[8 + FREE_VARS_OFFSET].item()
        fv2 = emb2[8 + FREE_VARS_OFFSET].item()
        result = backend.at_add(result, 8 + FREE_VARS_OFFSET, fv1 + fv2)

        bv1 = emb1[8 + BOUND_VARS_OFFSET].item()
        bv2 = emb2[8 + BOUND_VARS_OFFSET].item()
        result = backend.at_add(result, 8 + BOUND_VARS_OFFSET, bv1 + bv2)

        # Max predicate info
        pc1 = emb1[8 + PRED_COUNT_OFFSET].item()
        pc2 = emb2[8 + PRED_COUNT_OFFSET].item()
        result = backend.at_add(result, 8 + PRED_COUNT_OFFSET, pc1 + pc2)

        ma1 = emb1[8 + MAX_ARITY_OFFSET].item()
        ma2 = emb2[8 + MAX_ARITY_OFFSET].item()
        result = backend.at_add(result, 8 + MAX_ARITY_OFFSET, max(ma1, ma2))

        # Sentence only if both are sentences
        s1 = emb1[8 + SENTENCE_FLAG].item() > 0.5
        s2 = emb2[8 + SENTENCE_FLAG].item() > 0.5
        result = backend.at_add(result, 8 + SENTENCE_FLAG, 1.0 if (s1 and s2) else 0.0)

        return result

    def disjoin(self, emb1: Any, emb2: Any) -> Any:
        """Disjoin two formula embeddings (OR)."""
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # OR type
        result = backend.at_add(result, 8 + TYPE_OFFSET + 4, 1.0)

        # Max quantifier depth
        d1 = emb1[8 + QUANT_DEPTH_OFFSET].item()
        d2 = emb2[8 + QUANT_DEPTH_OFFSET].item()
        result = backend.at_add(result, 8 + QUANT_DEPTH_OFFSET, max(d1, d2))

        # Add complexities
        c1 = emb1[8 + COMPLEXITY_OFFSET].item()
        c2 = emb2[8 + COMPLEXITY_OFFSET].item()
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, c1 + c2 + 0.1)

        # Combine variables
        fv1 = emb1[8 + FREE_VARS_OFFSET].item()
        fv2 = emb2[8 + FREE_VARS_OFFSET].item()
        result = backend.at_add(result, 8 + FREE_VARS_OFFSET, fv1 + fv2)

        bv1 = emb1[8 + BOUND_VARS_OFFSET].item()
        bv2 = emb2[8 + BOUND_VARS_OFFSET].item()
        result = backend.at_add(result, 8 + BOUND_VARS_OFFSET, bv1 + bv2)

        # Max predicate info
        pc1 = emb1[8 + PRED_COUNT_OFFSET].item()
        pc2 = emb2[8 + PRED_COUNT_OFFSET].item()
        result = backend.at_add(result, 8 + PRED_COUNT_OFFSET, pc1 + pc2)

        ma1 = emb1[8 + MAX_ARITY_OFFSET].item()
        ma2 = emb2[8 + MAX_ARITY_OFFSET].item()
        result = backend.at_add(result, 8 + MAX_ARITY_OFFSET, max(ma1, ma2))

        # Sentence only if both are sentences
        s1 = emb1[8 + SENTENCE_FLAG].item() > 0.5
        s2 = emb2[8 + SENTENCE_FLAG].item() > 0.5
        result = backend.at_add(result, 8 + SENTENCE_FLAG, 1.0 if (s1 and s2) else 0.0)

        return result

    # =========================================================================
    # Substitution and Unification
    # =========================================================================

    def substitute_embedding(self, formula_emb: Any,
                              var_emb: Any,
                              term_emb: Any) -> Any:
        """
        Approximate substitution in embedding space.

        This is necessarily approximate since we lose term structure.

        Args:
            formula_emb: Embedding of the formula
            var_emb: Embedding representing the variable
            term_emb: Embedding of the replacement term

        Returns:
            Approximate embedding of the substituted formula
        """
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Copy most features from formula
        for i in range(10):
            result = backend.at_add(result, 8 + TYPE_OFFSET + i, formula_emb[8 + TYPE_OFFSET + i].item())

        result = backend.at_add(result, 8 + QUANT_DEPTH_OFFSET, formula_emb[8 + QUANT_DEPTH_OFFSET].item())
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, formula_emb[8 + COMPLEXITY_OFFSET].item())

        # Adjust free variables (one less after substitution if var was free)
        old_free = formula_emb[8 + FREE_VARS_OFFSET].item()
        result = backend.at_add(result, 8 + FREE_VARS_OFFSET, max(0, old_free - 0.1))

        result = backend.at_add(result, 8 + BOUND_VARS_OFFSET, formula_emb[8 + BOUND_VARS_OFFSET].item())
        result = backend.at_add(result, 8 + PRED_COUNT_OFFSET, formula_emb[8 + PRED_COUNT_OFFSET].item())
        result = backend.at_add(result, 8 + MAX_ARITY_OFFSET, formula_emb[8 + MAX_ARITY_OFFSET].item())

        # May become a sentence
        if old_free <= 0.1:
            result = backend.at_add(result, 8 + SENTENCE_FLAG, 1.0)

        return result

    def can_unify(self, emb1: Any, emb2: Any) -> bool:
        """
        Approximate check if two formulas might unify.

        This is a heuristic based on structural similarity.
        True unification requires the actual formula structure.

        Returns:
            True if unification might be possible
        """
        # Check if same domain
        if not self.is_valid(emb1) or not self.is_valid(emb2):
            return False

        # Check if same formula type
        type1 = -1
        type2 = -1
        max_val1 = 0.0
        max_val2 = 0.0
        for i in range(10):
            v1 = emb1[8 + TYPE_OFFSET + i].item()
            v2 = emb2[8 + TYPE_OFFSET + i].item()
            if v1 > max_val1:
                max_val1 = v1
                type1 = i
            if v2 > max_val2:
                max_val2 = v2
                type2 = i

        if type1 != type2:
            return False

        # Check if similar predicate structure
        pc1 = emb1[8 + PRED_COUNT_OFFSET].item()
        pc2 = emb2[8 + PRED_COUNT_OFFSET].item()
        if abs(pc1 - pc2) > 0.3:
            return False

        ma1 = emb1[8 + MAX_ARITY_OFFSET].item()
        ma2 = emb2[8 + MAX_ARITY_OFFSET].item()
        if abs(ma1 - ma2) > 0.3:
            return False

        return True

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def encode_predicate(self, name: str, arity: int) -> Any:
        """Encode a predicate with given arity (no specific terms)."""
        terms = [Term.variable(f"x{i}") for i in range(arity)]
        formula = FOLFormula.predicate(name, terms)
        return self.encode(formula)

    def encode_term(self, term: Term) -> Any:
        """
        Encode a term as part of an equality formula.

        Terms alone aren't formulas, so we encode t = t.
        """
        formula = FOLFormula.equals(term, term)
        return self.encode(formula)

    def get_quantifier_depth(self, emb: Any) -> float:
        """Get the quantifier depth from an embedding."""
        return emb[8 + QUANT_DEPTH_OFFSET].item() * 10.0

    def is_sentence(self, emb: Any) -> bool:
        """Check if embedding represents a sentence (closed formula)."""
        return emb[8 + SENTENCE_FLAG].item() > 0.5

    def get_free_var_count(self, emb: Any) -> int:
        """Get approximate number of free variables."""
        return int(emb[8 + FREE_VARS_OFFSET].item() * 10.0 + 0.5)

    def get_bound_var_count(self, emb: Any) -> int:
        """Get approximate number of bound variables."""
        return int(emb[8 + BOUND_VARS_OFFSET].item() * 10.0 + 0.5)
