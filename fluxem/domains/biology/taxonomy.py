"""
Taxonomy Encoder for Phylogenetics.

Encodes taxonomic classifications and phylogenetic relationships:
- Domain, Kingdom, Phylum, Class, Order, Family, Genus, Species
- Evolutionary distance
- Phylogenetic tree embeddings
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
# Taxonomic Data Structures
# =============================================================================

# Major taxonomic ranks
TAXONOMIC_RANKS = [
    "domain",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]

# Embedding layout within domain-specific region (64 dims starting at offset 8):
# dims 0-7:   Taxonomic rank codes (8 levels)
# dims 8-15:  Parent-child relationships
# dims 16-23: Phylogenetic distance vectors
# dims 24-31: Evolutionary divergence times
# dims 32-47: Morphological/genetic similarity scores
# dims 48-63: Phylogenetic tree position embedding

RANK_OFFSET = 0
RELATION_OFFSET = 8
DISTANCE_OFFSET = 16
DIVERGENCE_OFFSET = 24
SIMILARITY_OFFSET = 32
TREE_EMBED_OFFSET = 48


@dataclass
class Taxon:
    """Taxonomic classification."""

    domain: str = ""
    kingdom: str = ""
    phylum: str = ""
    class_: str = ""
    order: str = ""
    family: str = ""
    genus: str = ""
    species: str = ""
    common_name: Optional[str] = None


# =============================================================================
# Taxonomy Encoder
# =============================================================================


class TaxonomyEncoder:
    """
    Encoder for taxonomic classifications.

    Embedding captures:
    - Hierarchical taxonomic position
    - Phylogenetic relationships
    - Evolutionary distances
    - Morphological similarity
    """

    domain_tag = DOMAIN_TAGS["bio_taxonomy"]
    domain_name = "bio_taxonomy"

    def __init__(self):
        """Initialize taxonomy encoder with common taxa database."""
        self._taxon_index = {}
        self._build_taxon_database()

    def encode(self, taxon: Union[str, Taxon]) -> Any:
        """
        Encode a taxonomic classification to 128-dim embedding.

        Args:
            taxon: Taxon object or species name

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(taxon, str):
            taxon = self._get_taxon_from_name(taxon)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Encode each taxonomic rank
        ranks = [
            taxon.domain,
            taxon.kingdom,
            taxon.phylum,
            taxon.class_,
            taxon.order,
            taxon.family,
            taxon.genus,
            taxon.species,
        ]

        for i, rank_value in enumerate(ranks):
            if rank_value:
                idx = self._get_taxon_index(rank_value)
                emb = backend.at_add(emb, 8 + RANK_OFFSET + i, float(idx))

        # Encode parent-child relationships
        for i in range(7):  # 7 parent-child pairs
            parent_idx = self._get_taxon_index(ranks[i])
            child_idx = self._get_taxon_index(ranks[i + 1])
            if parent_idx > 0 and child_idx > 0:
                emb = backend.at_add(emb, 8 + RELATION_OFFSET + i, parent_idx)
                emb = backend.at_add(emb, 8 + RELATION_OFFSET + i + 8, child_idx)

        # Phylogenetic tree embedding
        tree_hash = self._phylogenetic_hash(taxon)
        for i, val in enumerate(tree_hash):
            emb = backend.at_add(emb, 8 + TREE_EMBED_OFFSET + i, val)

        return emb

    def decode(self, emb: Any) -> Taxon:
        """Decode embedding to taxonomic classification."""
        ranks = [""] * 8

        for i in range(8):
            idx = int(emb[8 + RANK_OFFSET + i].item())
            if idx > 0:
                ranks[i] = self._get_taxon_from_index(idx)

        return Taxon(
            domain=ranks[0],
            kingdom=ranks[1],
            phylum=ranks[2],
            class_=ranks[3],
            order=ranks[4],
            family=ranks[5],
            genus=ranks[6],
            species=ranks[7],
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid taxon."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Phylogenetic Operations
    # ========================================================================

    def common_ancestor(self, emb1: Any, emb2: Any) -> Any:
        """
        Find the most recent common ancestor (MRCA) of two taxa.
        """
        backend = get_backend()
        # Find lowest common rank
        for i in range(8):
            rank1_idx = emb1[8 + RANK_OFFSET + i].item()
            rank2_idx = emb2[8 + RANK_OFFSET + i].item()

            if rank1_idx == 0 or rank2_idx == 0:
                # No more data
                break

            if rank1_idx == rank2_idx:
                # Same taxon - common ancestor at this rank
                continue
            else:
                # Diverged - parent of i is MRCA
                if i > 0:
                    parent_idx = emb1[8 + RANK_OFFSET + i - 1].item()
                    result = create_embedding()
                    result = backend.at_add(result, slice(0, 8), self.domain_tag)
                    result = backend.at_add(result, 8 + RANK_OFFSET + i - 1, parent_idx)
                    return result
                else:
                    # Different domains - no common ancestor
                    return create_embedding()

        # One is ancestor of the other
        return emb1 if emb1[8 + RANK_OFFSET + 7].item() == 0 else emb2

    def phylogenetic_distance(self, emb1: Any, emb2: Any) -> float:
        """
        Calculate phylogenetic distance between two taxa.

        Based on taxonomic divergence.
        """
        # Count ranks where they differ
        distance = 0.0

        for i in range(8):
            rank1_idx = emb1[8 + RANK_OFFSET + i].item()
            rank2_idx = emb2[8 + RANK_OFFSET + i].item()

            if rank1_idx == 0 and rank2_idx == 0:
                # Both have no data - same species
                break
            elif rank1_idx != rank2_idx:
                # Different at this rank - higher weight for deeper divergence
                distance += 2.0 ** (7 - i)
            elif rank1_idx == rank2_idx:
                # Same - continue
                continue

        return distance

    def is_related(self, emb1: Any, emb2: Any, min_rank: int = 5) -> bool:
        """
        Check if two taxa are related at or above a given rank.

        Args:
            emb1: First taxon embedding
            emb2: Second taxon embedding
            min_rank: Minimum rank to match (0=domain, 7=species)
        """
        for i in range(min_rank):
            rank1_idx = emb1[8 + RANK_OFFSET + i].item()
            rank2_idx = emb2[8 + RANK_OFFSET + i].item()

            if rank1_idx != rank2_idx:
                return False

        return True

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_taxon_index(self, name: str) -> int:
        """Get or create index for a taxon name."""
        if not name:
            return 0
        if name not in self._taxon_index:
            self._taxon_index[name] = len(self._taxon_index) + 1
        return self._taxon_index[name]

    def _get_taxon_from_index(self, idx: int) -> str:
        """Reverse lookup for taxon index."""
        if idx == 0:
            return ""
        for name, i in self._taxon_index.items():
            if i == idx:
                return name
        return ""

    def _get_taxon_from_name(self, name: str) -> Taxon:
        """Get Taxon object from common name."""
        # Check if it's already a scientific name (Genus species)
        if " " in name:
            parts = name.split()
            if len(parts) >= 2:
                return Taxon(genus=parts[0], species=" ".join(parts[1:]))

        # Look up in common taxa database
        common_taxa = {
            "human": Taxon(
                domain="Eukarya",
                kingdom="Animalia",
                phylum="Chordata",
                class_="Mammalia",
                order="Primates",
                family="Hominidae",
                genus="Homo",
                species="sapiens",
                common_name="human",
            ),
            "mouse": Taxon(
                domain="Eukarya",
                kingdom="Animalia",
                phylum="Chordata",
                class_="Mammalia",
                order="Rodentia",
                family="Muridae",
                genus="Mus",
                species="musculus",
                common_name="mouse",
            ),
            "yeast": Taxon(
                domain="Eukarya",
                kingdom="Fungi",
                phylum="Ascomycota",
                class_="Saccharomycetes",
                order="Saccharomycetales",
                family="Saccharomycetaceae",
                genus="Saccharomyces",
                species="cerevisiae",
                common_name="yeast",
            ),
            "e coli": Taxon(
                domain="Bacteria",
                kingdom="Bacteria",
                phylum="Proteobacteria",
                class_="Gammaproteobacteria",
                order="Enterobacterales",
                family="Enterobacteriaceae",
                genus="Escherichia",
                species="coli",
                common_name="E. coli",
            ),
        }

        return common_taxa.get(name.lower(), Taxon())

    def _phylogenetic_hash(self, taxon: Taxon) -> List[float]:
        """Create phylogenetic tree hash embedding."""
        values = []
        ranks = [
            taxon.domain,
            taxon.kingdom,
            taxon.phylum,
            taxon.class_,
            taxon.order,
            taxon.family,
            taxon.genus,
            taxon.species,
        ]

        for i in range(16):
            val = 0.0
            for j, rank in enumerate(ranks):
                if rank:
                    val += hash(rank) * ((i + 1) ** j)
            values.append((val % 100000) / 100000.0)

        return values

    def _build_taxon_database(self):
        """Build database of common taxa."""
        # Add common taxa
        common_taxa = [
            "Eukarya",
            "Bacteria",
            "Archaea",
            "Animalia",
            "Plantae",
            "Fungi",
            "Protista",
            "Chordata",
            "Arthropoda",
            "Mollusca",
            "Mammalia",
            "Insecta",
            "Aves",
            "Primates",
            "Rodentia",
            "Carnivora",
            "Hominidae",
            "Muridae",
            "Felidae",
            "Homo",
            "Mus",
            "Felis",
            "sapiens",
            "musculus",
            "catus",
        ]

        for taxon in common_taxa:
            self._get_taxon_index(taxon)


# =============================================================================
# Utility Functions
# =============================================================================


def create_taxonomy_tree(
    taxa: List[str], encoder: TaxonomyEncoder
) -> Dict[str, List[str]]:
    """
    Create a simple taxonomy tree from a list of species names.

    Returns nested dict representing tree structure.
    """
    emb_taxa = [(name, encoder.encode(name)) for name in taxa]

    tree = {}
    for name, emb in emb_taxa:
        taxon = encoder.decode(emb)

        # Build path in tree
        current = tree
        ranks = [
            taxon.domain,
            taxon.kingdom,
            taxon.phylum,
            taxon.class_,
            taxon.order,
            taxon.family,
            taxon.genus,
            taxon.species,
        ]

        for i, rank in enumerate(ranks):
            if rank:
                if rank not in current:
                    current[rank] = {}
                if i == 7:  # Species level
                    current[rank][name] = None
                current = current[rank]

    return tree
