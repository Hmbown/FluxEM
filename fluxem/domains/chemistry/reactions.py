"""
Reaction Encoder for Chemistry.

Embeds chemical reactions for stoichiometric analysis.
The key operation is deterministic balance checking.

Stoichiometry is integer linear algebra:
- Conservation of mass = sum of reactants equals sum of products
- Balancing = finding integer coefficients that satisfy conservation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
)
from .molecules import MoleculeEncoder, Formula, COMMON_ELEMENTS, ELEMENT_COUNTS_OFFSET


# =============================================================================
# Reaction Data Structure
# =============================================================================

@dataclass
class Reaction:
    """
    Chemical reaction with reactants and products.

    Each side is a list of (coefficient, formula) pairs.

    Example:
        2H2 + O2 -> 2H2O

        Reaction(
            reactants=[(2, Formula({'H': 2})), (1, Formula({'O': 2}))],
            products=[(2, Formula({'H': 2, 'O': 1}))]
        )
    """
    reactants: List[Tuple[int, Formula]] = field(default_factory=list)
    products: List[Tuple[int, Formula]] = field(default_factory=list)

    @classmethod
    def parse(cls, reaction_str: str) -> 'Reaction':
        """
        Parse a reaction string.

        Format: "2H2 + O2 -> 2H2O" or "2H2 + O2 = 2H2O"
        """
        # Split into reactants and products
        if '->' in reaction_str:
            left, right = reaction_str.split('->')
        elif '=' in reaction_str:
            left, right = reaction_str.split('=')
        else:
            raise ValueError("Reaction must contain '->' or '='")

        def parse_side(side: str) -> List[Tuple[int, Formula]]:
            terms = []
            for term in side.split('+'):
                term = term.strip()
                if not term:
                    continue

                # Extract coefficient
                coeff = 1
                i = 0
                while i < len(term) and term[i].isdigit():
                    i += 1
                if i > 0:
                    coeff = int(term[:i])
                    term = term[i:]

                formula = Formula.parse(term)
                terms.append((coeff, formula))
            return terms

        return cls(
            reactants=parse_side(left),
            products=parse_side(right)
        )

    def reactant_composition(self) -> Dict[str, int]:
        """Total element counts on reactant side."""
        total = {}
        for coeff, formula in self.reactants:
            for symbol, count in formula.composition.items():
                total[symbol] = total.get(symbol, 0) + coeff * count
        return total

    def product_composition(self) -> Dict[str, int]:
        """Total element counts on product side."""
        total = {}
        for coeff, formula in self.products:
            for symbol, count in formula.composition.items():
                total[symbol] = total.get(symbol, 0) + coeff * count
        return total

    def is_balanced(self) -> bool:
        """Check if the reaction is balanced (mass conservation)."""
        return self.reactant_composition() == self.product_composition()

    def imbalance(self) -> Dict[str, int]:
        """
        Get the imbalance for each element.

        Positive = excess in products, Negative = excess in reactants
        """
        reactants = self.reactant_composition()
        products = self.product_composition()

        all_elements = set(reactants.keys()) | set(products.keys())
        imbalance = {}

        for elem in all_elements:
            diff = products.get(elem, 0) - reactants.get(elem, 0)
            if diff != 0:
                imbalance[elem] = diff

        return imbalance

    def __repr__(self) -> str:
        def format_side(terms):
            parts = []
            for coeff, formula in terms:
                if coeff == 1:
                    parts.append(str(formula))
                else:
                    parts.append(f"{coeff}{formula}")
            return " + ".join(parts)

        return f"{format_side(self.reactants)} -> {format_side(self.products)}"


# =============================================================================
# Reaction Encoder
# =============================================================================

class ReactionEncoder:
    """
    Encoder for chemical reactions.

    Embedding structure uses the full 128 dims:
        dims 0-7:    Domain tag
        dims 8-71:   Reactant side composition (same as molecule encoding)
        dims 72-127: Product side composition (56 dims)

    This enables deterministic balance checking by comparing element counts
    between the two halves of the embedding.
    """

    domain_tag = DOMAIN_TAGS["chem_reaction"]
    domain_name = "chem_reaction"

    def __init__(self):
        self.mol_encoder = MoleculeEncoder()

    def encode(self, reaction: Union[str, Reaction]) -> Any:
        """
        Encode a chemical reaction.

        Args:
            reaction: Reaction string or object

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(reaction, str):
            reaction = Reaction.parse(reaction)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Encode reactant side composition (dims 8-71)
        reactant_comp = reaction.reactant_composition()
        emb = self._encode_composition(emb, reactant_comp, offset=8)

        # Encode product side composition (dims 72-127)
        product_comp = reaction.product_composition()
        emb = self._encode_composition(emb, product_comp, offset=72)

        return emb

    def _encode_composition(
        self,
        emb: Any,
        composition: Dict[str, int],
        offset: int
    ) -> Any:
        """Encode element composition at specified offset."""
        # Use same layout as molecule encoder for element counts
        for i, symbol in enumerate(COMMON_ELEMENTS):
            if i >= 16:  # Only 16 elements fit in 32 dims
                break

            count = composition.get(symbol, 0)
            elem_offset = offset + i * 2

            if count > 0:
                emb = backend.at_add(emb, elem_offset, 1.0)
                emb = backend.at_add(emb, elem_offset + 1, backend.log(backend.array(float(count + 1))))

        return emb

    def _decode_composition(self, emb: Any, offset: int) -> Dict[str, int]:
        """Decode element composition from specified offset."""
        backend = get_backend()
        composition = {}

        for i, symbol in enumerate(COMMON_ELEMENTS):
            if i >= 16:
                break

            elem_offset = offset + i * 2
            present = emb[elem_offset].item() > 0.5

            if present:
                log_count = emb[elem_offset + 1].item()
                count = int(round(backend.exp(backend.array(log_count)).item() - 1))
                if count > 0:
                    composition[symbol] = count

        return composition

    def decode_reactant_composition(self, emb: Any) -> Dict[str, int]:
        """Decode the reactant side composition."""
        return self._decode_composition(emb, offset=8)

    def decode_product_composition(self, emb: Any) -> Dict[str, int]:
        """Decode the product side composition."""
        return self._decode_composition(emb, offset=72)

    def decode(self, emb: Any) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Decode embedding to (reactant_composition, product_composition).

        Note: Original coefficients are lost - only total counts preserved.
        """
        return (
            self.decode_reactant_composition(emb),
            self.decode_product_composition(emb)
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid reaction."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Balance checking
    # =========================================================================

    def is_balanced(self, emb: Any) -> bool:
        """
        Check if the reaction is balanced (mass conservation).
        """
        backend = get_backend()
        # Compare element counts between reactant and product sides
        for i in range(min(16, len(COMMON_ELEMENTS))):
            reactant_offset = 8 + i * 2
            product_offset = 72 + i * 2

            # Check presence flags
            r_present = emb[reactant_offset].item() > 0.5
            p_present = emb[product_offset].item() > 0.5

            if r_present != p_present:
                return False

            if r_present:
                r_count = round(backend.exp(emb[reactant_offset + 1]).item() - 1)
                p_count = round(backend.exp(emb[product_offset + 1]).item() - 1)

                if abs(r_count - p_count) > 0.5:
                    return False

        return True

    def get_imbalance(self, emb: Any) -> Dict[str, int]:
        """
        Get the imbalance for each element.

        Returns dict mapping element -> (products - reactants)
        Positive = excess in products, Negative = excess in reactants
        Empty dict means balanced.
        """
        backend = get_backend()
        imbalance = {}

        for i, symbol in enumerate(COMMON_ELEMENTS):
            if i >= 16:
                break

            reactant_offset = 8 + i * 2
            product_offset = 72 + i * 2

            r_count = 0
            p_count = 0

            if emb[reactant_offset].item() > 0.5:
                r_count = int(round(backend.exp(emb[reactant_offset + 1]).item() - 1))

            if emb[product_offset].item() > 0.5:
                p_count = int(round(backend.exp(emb[product_offset + 1]).item() - 1))

            diff = p_count - r_count
            if diff != 0:
                imbalance[symbol] = diff

        return imbalance

    def balance_check_message(self, emb: Any) -> str:
        """Get a human-readable balance check message."""
        if self.is_balanced(emb):
            return "Reaction is balanced."

        imbalance = self.get_imbalance(emb)
        parts = []
        for elem, diff in imbalance.items():
            if diff > 0:
                parts.append(f"{elem}: {diff} excess in products")
            else:
                parts.append(f"{elem}: {-diff} excess in reactants")

        return "Reaction is NOT balanced: " + ", ".join(parts)

    # =========================================================================
    # Reaction Operations
    # =========================================================================

    def combine_reactions(self, emb1: Any, emb2: Any) -> Any:
        """
        Combine two reactions (Hess's law).

        Adds both sides of both reactions.
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Add reactant sides
        for i in range(min(16, len(COMMON_ELEMENTS))):
            offset = 8 + i * 2

            r1 = 0
            r2 = 0
            if emb1[offset].item() > 0.5:
                r1 = int(round(backend.exp(emb1[offset + 1]).item() - 1))
            if emb2[offset].item() > 0.5:
                r2 = int(round(backend.exp(emb2[offset + 1]).item() - 1))

            total = r1 + r2
            if total > 0:
                result = backend.at_add(result, offset, 1.0)
                result = backend.at_add(result, offset + 1, backend.log(backend.array(float(total + 1))))

        # Add product sides
        for i in range(min(16, len(COMMON_ELEMENTS))):
            offset = 72 + i * 2

            p1 = 0
            p2 = 0
            if emb1[offset].item() > 0.5:
                p1 = int(round(backend.exp(emb1[offset + 1]).item() - 1))
            if emb2[offset].item() > 0.5:
                p2 = int(round(backend.exp(emb2[offset + 1]).item() - 1))

            total = p1 + p2
            if total > 0:
                result = backend.at_add(result, offset, 1.0)
                result = backend.at_add(result, offset + 1, backend.log(backend.array(float(total + 1))))

        return result

    def reverse(self, emb: Any) -> Any:
        """
        Reverse a reaction (swap reactants and products).
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Swap: reactants (8-71) <-> products (72-127)
        for i in range(min(16, len(COMMON_ELEMENTS))):
            r_offset = 8 + i * 2
            p_offset = 72 + i * 2

            # Copy product to reactant position
            result = backend.at_add(result, r_offset, emb[p_offset])
            result = backend.at_add(result, r_offset + 1, emb[p_offset + 1])

            # Copy reactant to product position
            result = backend.at_add(result, p_offset, emb[r_offset])
            result = backend.at_add(result, p_offset + 1, emb[r_offset + 1])

        return result

    def scale_reaction(self, emb: Any, n: int) -> Any:
        """
        Scale all coefficients in a reaction by n.
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Scale both sides
        for i in range(min(16, len(COMMON_ELEMENTS))):
            for base_offset in [8, 72]:  # Reactants and products
                offset = base_offset + i * 2

                if emb[offset].item() > 0.5:
                    count = int(round(backend.exp(emb[offset + 1]).item() - 1))
                    scaled = count * n
                    if scaled > 0:
                        result = backend.at_add(result, offset, 1.0)
                        result = backend.at_add(result, offset + 1, 
                            backend.log(backend.array(float(scaled + 1)))
                        )

        return result
