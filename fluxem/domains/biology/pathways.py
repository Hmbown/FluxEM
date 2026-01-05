"""
Metabolic Pathway Encoder for Biochemistry.

Encodes metabolic pathways and biochemical reactions:
- Stoichiometric representation of reactions
- Enzyme-catalyzed processes
- Thermodynamics (ΔG, Keq)
- Pathway topology and flux
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
    log_encode_value,
    log_decode_value,
)

# =============================================================================
# Biochemical Data Structures
# =============================================================================


@dataclass
class BiochemicalReaction:
    """Biochemical reaction with enzyme information."""

    reactants: Dict[str, float]  # stoichiometric coefficients
    products: Dict[str, float]  # stoichiometric coefficients
    enzyme: Optional[str] = None  # EC number or enzyme name
    delta_g: Optional[float] = None  # Free energy change (kJ/mol)
    reversible: bool = True


# Embedding layout within domain-specific region (64 dims starting at offset 8):
# dims 0-15:  Reactant stoichiometry (up to 16 metabolites)
# dims 16-31: Product stoichiometry (up to 16 metabolites)
# dim 32:     Reaction directionality (-1: reverse, 0: reversible, 1: forward)
# dim 33:     ΔG (log encoded)
# dim 34:     Equilibrium constant (log encoded)
# dims 35-39: Enzyme information (EC number components)
# dims 40-47: Pathway context (pathway ID, position, etc.)
# dims 48-63: Metabolite fingerprints

REACTANT_OFFSET = 0
PRODUCT_OFFSET = 16
DIRECTION_OFFSET = 32
DG_OFFSET = 33
KEQ_OFFSET = 34
ENZYME_OFFSET = 35
PATHWAY_OFFSET = 40
METABOLITE_OFFSET = 48


# =============================================================================
# Metabolic Pathway Encoder
# =============================================================================


class PathwayEncoder:
    """
    Encoder for metabolic pathways and biochemical reactions.

    Embedding captures:
    - Stoichiometry (integer counts)
    - Thermodynamics
    - Enzyme specificity
    - Pathway context
    """

    domain_tag = DOMAIN_TAGS["bio_pathway"]
    domain_name = "bio_pathway"

    def __init__(self):
        """Initialize pathway encoder."""
        self._metabolite_index = {}
        self._enzyme_index = {}

    def encode_reaction(self, reaction: BiochemicalReaction) -> Any:
        """
        Encode a biochemical reaction to 128-dim embedding.

        Args:
            reaction: BiochemicalReaction object

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Reactants stoichiometry
        for i, (met, coeff) in enumerate(list(reaction.reactants.items())[:16]):
            idx = self._get_metabolite_index(met)
            emb = backend.at_add(emb, 8 + REACTANT_OFFSET + i, float(idx))
            emb = backend.at_add(emb, 8 + REACTANT_OFFSET + i + 16, coeff)

        # Products stoichiometry
        for i, (met, coeff) in enumerate(list(reaction.products.items())[:16]):
            idx = self._get_metabolite_index(met)
            emb = backend.at_add(emb, 8 + PRODUCT_OFFSET + i, float(idx))
            emb = backend.at_add(emb, 8 + PRODUCT_OFFSET + i + 16, coeff)

        # Directionality
        if not reaction.reversible:
            emb = backend.at_add(emb, 8 + DIRECTION_OFFSET, 1.0)
        else:
            emb = backend.at_add(emb, 8 + DIRECTION_OFFSET, 0.0)

        # ΔG (free energy change)
        if reaction.delta_g is not None:
            _, log_dg = log_encode_value(abs(reaction.delta_g))
            emb = backend.at_add(emb, 8 + DG_OFFSET, log_dg)

        # Enzyme information
        if reaction.enzyme:
            self._encode_enzyme(emb, reaction.enzyme)

        return emb

    def encode_pathway(
        self, reactions: List[BiochemicalReaction], pathway_id: int = 0
    ) -> Any:
        """
        Encode an entire metabolic pathway.

        Args:
            reactions: List of reactions in the pathway
            pathway_id: Optional pathway identifier

        Returns:
            128-dim embedding
        """
        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Pathway ID
        emb = backend.at_add(emb, 8 + PATHWAY_OFFSET, float(pathway_id))

        # Aggregate stoichiometry from all reactions
        total_reactants = {}
        total_products = {}

        for rxn in reactions:
            for met, coeff in rxn.reactants.items():
                total_reactants[met] = total_reactants.get(met, 0) + coeff
            for met, coeff in rxn.products.items():
                total_products[met] = total_products.get(met, 0) + coeff

        # Encode aggregated stoichiometry
        for i, (met, coeff) in enumerate(list(total_reactants.items())[:16]):
            idx = self._get_metabolite_index(met)
            emb = backend.at_add(emb, 8 + REACTANT_OFFSET + i, float(idx))
            emb = backend.at_add(emb, 8 + REACTANT_OFFSET + i + 16, coeff)

        for i, (met, coeff) in enumerate(list(total_products.items())[:16]):
            idx = self._get_metabolite_index(met)
            emb = backend.at_add(emb, 8 + PRODUCT_OFFSET + i, float(idx))
            emb = backend.at_add(emb, 8 + PRODUCT_OFFSET + i + 16, coeff)

        return emb

    def decode(self, emb: Any) -> BiochemicalReaction:
        """Decode embedding to biochemical reaction."""
        # Extract reactants and products
        reactants = {}
        products = {}

        for i in range(16):
            met_idx = int(emb[8 + REACTANT_OFFSET + i].item())
            coeff = emb[8 + REACTANT_OFFSET + i + 16].item()
            if coeff > 0:
                met = self._get_metabolite_from_index(met_idx)
                reactants[met] = coeff

            met_idx = int(emb[8 + PRODUCT_OFFSET + i].item())
            coeff = emb[8 + PRODUCT_OFFSET + i + 16].item()
            if coeff > 0:
                met = self._get_metabolite_from_index(met_idx)
                products[met] = coeff

        return BiochemicalReaction(
            reactants=reactants,
            products=products,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid pathway/reaction."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Algebraic Operations
    # ========================================================================

    def add_reactions(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two reactions (combine them).
        """
        backend = get_backend()
        result = emb1 + emb2

        # Renormalize domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag - result[0:8])

        return result

    def subtract_reactions(self, emb1: Any, emb2: Any) -> Any:
        """
        Subtract one reaction from another.

        Useful for finding net reactions in pathways.
        """
        backend = get_backend()
        result = emb1 - emb2

        # Renormalize domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag - result[0:8])

        return result

    def check_balance(self, emb: Any) -> bool:
        """
        Check if a reaction is mass-balanced.

        Compares total stoichiometry of reactants vs products.
        """
        backend = get_backend()
        # Extract reactant and product stoichiometry
        reactant_coeffs = emb[8 + REACTANT_OFFSET + 16 : 8 + REACTANT_OFFSET + 32]
        product_coeffs = emb[8 + PRODUCT_OFFSET + 16 : 8 + PRODUCT_OFFSET + 32]

        # Simple check: total moles in = total moles out
        reactant_total = backend.sum(reactant_coeffs).item()
        product_total = backend.sum(product_coeffs).item()

        return abs(reactant_total - product_total) < 0.01

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_metabolite_index(self, metabolite: str) -> int:
        """Get or create index for a metabolite."""
        if metabolite not in self._metabolite_index:
            self._metabolite_index[metabolite] = len(self._metabolite_index)
        return self._metabolite_index[metabolite]

    def _get_metabolite_from_index(self, idx: int) -> str:
        """Reverse lookup for metabolite index."""
        for met, i in self._metabolite_index.items():
            if i == idx:
                return met
        return f"M{idx}"

    def _encode_enzyme(self, emb: Any, enzyme: str) -> Any:
        """Encode enzyme information."""
        backend = get_backend()
        # Parse EC number if present
        if enzyme.startswith("EC "):
            ec_parts = enzyme[3:].split(".")
            for i, part in enumerate(ec_parts[:5]):
                try:
                    ec_val = float(part) / 99.0  # Normalize EC number components
                    emb = backend.at_add(emb, 8 + ENZYME_OFFSET + i, ec_val)
                except ValueError:
                    pass
        else:
            # Hash enzyme name
            hash_val = hash(enzyme) % 10000 / 10000.0
            emb = backend.at_add(emb, 8 + ENZYME_OFFSET, hash_val)

        return emb


# =============================================================================
# Common Metabolic Pathways (Examples)
# =============================================================================

GLYCOLYSIS = [
    BiochemicalReaction(
        reactants={"Glucose": 1, "ATP": 1},
        products={"Glucose-6-phosphate": 1, "ADP": 1},
        enzyme="EC 2.7.1.1",  # Hexokinase
    ),
    BiochemicalReaction(
        reactants={"Pyruvate": 1, "NAD+": 1},
        products={"Acetyl-CoA": 1, "CO2": 1, "NADH": 1},
        enzyme="EC 1.2.4.1",  # Pyruvate dehydrogenase
    ),
    BiochemicalReaction(
        reactants={"Acetyl-CoA": 1, "Oxaloacetate": 1, "H2O": 1},
        products={"Citrate": 1, "CoA": 1},
        enzyme="EC 2.3.3.1",  # Citrate synthase
    ),
]

KREBS_CYCLE = [
    BiochemicalReaction(
        reactants={"Citrate": 1},
        products={"Isocitrate": 1},
        enzyme="EC 4.2.1.3",  # Aconitase
    ),
    BiochemicalReaction(
        reactants={"Isocitrate": 1, "NAD+": 1},
        products={"Alpha-ketoglutarate": 1, "CO2": 1, "NADH": 1},
        enzyme="EC 1.1.1.41",  # Isocitrate dehydrogenase
    ),
]


def calculate_net_reaction(reactions: List[BiochemicalReaction]) -> BiochemicalReaction:
    """
    Calculate the net reaction from a list of reactions.

    Useful for finding overall stoichiometry of pathways.
    """
    net_reactants = {}
    net_products = {}

    for rxn in reactions:
        for met, coeff in rxn.reactants.items():
            net_reactants[met] = net_reactants.get(met, 0) + coeff
        for met, coeff in rxn.products.items():
            net_products[met] = net_products.get(met, 0) + coeff

    # Cancel out metabolites that appear on both sides
    all_metabolites = set(net_reactants.keys()) | set(net_products.keys())
    for met in all_metabolites:
        reactant_coeff = net_reactants.get(met, 0)
        product_coeff = net_products.get(met, 0)

        if reactant_coeff > product_coeff:
            net_reactants[met] = reactant_coeff - product_coeff
            if met in net_products:
                del net_products[met]
        elif product_coeff > reactant_coeff:
            net_products[met] = product_coeff - reactant_coeff
            if met in net_reactants:
                del net_reactants[met]
        else:
            if met in net_reactants:
                del net_reactants[met]
            if met in net_products:
                del net_products[met]

    return BiochemicalReaction(
        reactants=net_reactants,
        products=net_products,
    )
