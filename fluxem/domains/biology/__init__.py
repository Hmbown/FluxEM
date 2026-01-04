"""
Biology Domain Encoders for FluxEM.

Provides algebraic embeddings for biological entities and processes.
"""

from .dna import DNAEncoder, complement, reverse_complement, translate_dna_to_protein
from .protein import ProteinEncoder
from .pathways import PathwayEncoder, BiochemicalReaction, calculate_net_reaction
from .taxonomy import TaxonomyEncoder, Taxon

__all__ = [
    "DNAEncoder",
    "ProteinEncoder",
    "PathwayEncoder",
    "TaxonomyEncoder",
    "BiochemicalReaction",
    "Taxon",
    "complement",
    "reverse_complement",
    "translate_dna_to_protein",
    "calculate_net_reaction",
]
