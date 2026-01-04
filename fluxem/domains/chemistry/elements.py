"""
Element Encoder for Chemistry.

Embeds chemical elements with their properties from the periodic table.
This is a lookup-based encoder - properties are deterministic, not learned.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


# =============================================================================
# Element Data Structure
# =============================================================================

@dataclass
class Element:
    """Chemical element with its properties."""
    symbol: str
    name: str
    atomic_number: int
    atomic_mass: float
    period: int
    group: int
    block: str  # 's', 'p', 'd', 'f'
    valence_electrons: int
    electronegativity: Optional[float]  # Pauling scale
    common_oxidation_states: List[int]
    electron_config: str
    category: str  # e.g., 'alkali metal', 'noble gas', etc.


# Embedding layout within domain-specific region (64 dims starting at offset 8):
# dims 0-1:   Atomic number (1, log(Z))
# dims 2-3:   Atomic mass (1, log(mass))
# dims 4-7:   Position [period, group/10, block_code, valence]
# dims 8-9:   Electronegativity (exists_flag, value/4)
# dims 10-13: Oxidation states (up to 4, normalized)
# dims 14-15: Category code
# dims 16-31: Reserved/property features

ATOMIC_NUM_OFFSET = 0
ATOMIC_MASS_OFFSET = 2
POSITION_OFFSET = 4
ELECTRONEGATIVITY_OFFSET = 8
OXIDATION_OFFSET = 10
CATEGORY_OFFSET = 14

# Block encoding
BLOCK_CODES = {'s': 0.0, 'p': 1.0, 'd': 2.0, 'f': 3.0}

# Category encoding
CATEGORY_CODES = {
    'alkali metal': 0,
    'alkaline earth metal': 1,
    'transition metal': 2,
    'post-transition metal': 3,
    'metalloid': 4,
    'nonmetal': 5,
    'halogen': 6,
    'noble gas': 7,
    'lanthanide': 8,
    'actinide': 9,
}


# =============================================================================
# Periodic Table Data
# =============================================================================

# Complete periodic table data (first 118 elements)
PERIODIC_TABLE: Dict[str, Element] = {}

def _init_periodic_table():
    """Initialize the periodic table data."""
    global PERIODIC_TABLE

    # Data for common elements (abbreviated for brevity - full table in data file)
    elements_data = [
        # symbol, name, Z, mass, period, group, block, valence, EN, oxidation, config, category
        ("H", "Hydrogen", 1, 1.008, 1, 1, 's', 1, 2.20, [1, -1], "1s1", "nonmetal"),
        ("He", "Helium", 2, 4.003, 1, 18, 's', 2, None, [0], "1s2", "noble gas"),
        ("Li", "Lithium", 3, 6.941, 2, 1, 's', 1, 0.98, [1], "[He] 2s1", "alkali metal"),
        ("Be", "Beryllium", 4, 9.012, 2, 2, 's', 2, 1.57, [2], "[He] 2s2", "alkaline earth metal"),
        ("B", "Boron", 5, 10.81, 2, 13, 'p', 3, 2.04, [3], "[He] 2s2 2p1", "metalloid"),
        ("C", "Carbon", 6, 12.01, 2, 14, 'p', 4, 2.55, [-4, 4], "[He] 2s2 2p2", "nonmetal"),
        ("N", "Nitrogen", 7, 14.01, 2, 15, 'p', 5, 3.04, [-3, 3, 5], "[He] 2s2 2p3", "nonmetal"),
        ("O", "Oxygen", 8, 16.00, 2, 16, 'p', 6, 3.44, [-2], "[He] 2s2 2p4", "nonmetal"),
        ("F", "Fluorine", 9, 19.00, 2, 17, 'p', 7, 3.98, [-1], "[He] 2s2 2p5", "halogen"),
        ("Ne", "Neon", 10, 20.18, 2, 18, 'p', 8, None, [0], "[He] 2s2 2p6", "noble gas"),
        ("Na", "Sodium", 11, 22.99, 3, 1, 's', 1, 0.93, [1], "[Ne] 3s1", "alkali metal"),
        ("Mg", "Magnesium", 12, 24.31, 3, 2, 's', 2, 1.31, [2], "[Ne] 3s2", "alkaline earth metal"),
        ("Al", "Aluminum", 13, 26.98, 3, 13, 'p', 3, 1.61, [3], "[Ne] 3s2 3p1", "post-transition metal"),
        ("Si", "Silicon", 14, 28.09, 3, 14, 'p', 4, 1.90, [-4, 4], "[Ne] 3s2 3p2", "metalloid"),
        ("P", "Phosphorus", 15, 30.97, 3, 15, 'p', 5, 2.19, [-3, 3, 5], "[Ne] 3s2 3p3", "nonmetal"),
        ("S", "Sulfur", 16, 32.07, 3, 16, 'p', 6, 2.58, [-2, 4, 6], "[Ne] 3s2 3p4", "nonmetal"),
        ("Cl", "Chlorine", 17, 35.45, 3, 17, 'p', 7, 3.16, [-1, 1, 3, 5, 7], "[Ne] 3s2 3p5", "halogen"),
        ("Ar", "Argon", 18, 39.95, 3, 18, 'p', 8, None, [0], "[Ne] 3s2 3p6", "noble gas"),
        ("K", "Potassium", 19, 39.10, 4, 1, 's', 1, 0.82, [1], "[Ar] 4s1", "alkali metal"),
        ("Ca", "Calcium", 20, 40.08, 4, 2, 's', 2, 1.00, [2], "[Ar] 4s2", "alkaline earth metal"),
        ("Fe", "Iron", 26, 55.85, 4, 8, 'd', 2, 1.83, [2, 3], "[Ar] 3d6 4s2", "transition metal"),
        ("Cu", "Copper", 29, 63.55, 4, 11, 'd', 1, 1.90, [1, 2], "[Ar] 3d10 4s1", "transition metal"),
        ("Zn", "Zinc", 30, 65.38, 4, 12, 'd', 2, 1.65, [2], "[Ar] 3d10 4s2", "transition metal"),
        ("Br", "Bromine", 35, 79.90, 4, 17, 'p', 7, 2.96, [-1, 1, 3, 5], "[Ar] 3d10 4s2 4p5", "halogen"),
        ("I", "Iodine", 53, 126.90, 5, 17, 'p', 7, 2.66, [-1, 1, 3, 5, 7], "[Kr] 4d10 5s2 5p5", "halogen"),
        ("Au", "Gold", 79, 196.97, 6, 11, 'd', 1, 2.54, [1, 3], "[Xe] 4f14 5d10 6s1", "transition metal"),
        ("Hg", "Mercury", 80, 200.59, 6, 12, 'd', 2, 2.00, [1, 2], "[Xe] 4f14 5d10 6s2", "transition metal"),
        ("Pb", "Lead", 82, 207.2, 6, 14, 'p', 4, 2.33, [2, 4], "[Xe] 4f14 5d10 6s2 6p2", "post-transition metal"),
        ("U", "Uranium", 92, 238.03, 7, 3, 'f', 2, 1.38, [3, 4, 5, 6], "[Rn] 5f3 6d1 7s2", "actinide"),
    ]

    for data in elements_data:
        symbol, name, Z, mass, period, group, block, valence, en, ox, config, cat = data
        PERIODIC_TABLE[symbol] = Element(
            symbol=symbol,
            name=name,
            atomic_number=Z,
            atomic_mass=mass,
            period=period,
            group=group,
            block=block,
            valence_electrons=valence,
            electronegativity=en,
            common_oxidation_states=ox,
            electron_config=config,
            category=cat,
        )

    # Also index by atomic number
    for symbol, elem in list(PERIODIC_TABLE.items()):
        PERIODIC_TABLE[elem.atomic_number] = elem

_init_periodic_table()


# =============================================================================
# Element Encoder
# =============================================================================

class ElementEncoder:
    """
    Encoder for chemical elements.

    This is a lookup-based encoder - all properties come from the
    periodic table, not from learning.
    """

    domain_tag = DOMAIN_TAGS["chem_element"]
    domain_name = "chem_element"

    def encode(self, element: Union[str, int, Element]) -> Any:
        """
        Encode an element to 128-dim embedding.

        Args:
            element: Symbol (str), atomic number (int), or Element object

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(element, str):
            elem = PERIODIC_TABLE.get(element)
            if elem is None:
                raise ValueError(f"Unknown element symbol: {element}")
        elif isinstance(element, int):
            elem = PERIODIC_TABLE.get(element)
            if elem is None:
                raise ValueError(f"Unknown atomic number: {element}")
        else:
            elem = element

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Atomic number (log encoded)
        emb = backend.at_add(emb, 8 + ATOMIC_NUM_OFFSET, 1.0)
        emb = backend.at_add(emb, 8 + ATOMIC_NUM_OFFSET + 1, backend.log(backend.array(float(elem.atomic_number))))

        # Atomic mass (log encoded)
        emb = backend.at_add(emb, 8 + ATOMIC_MASS_OFFSET, 1.0)
        emb = backend.at_add(emb, 8 + ATOMIC_MASS_OFFSET + 1, backend.log(backend.array(elem.atomic_mass)))

        # Position in periodic table
        emb = backend.at_add(emb, 8 + POSITION_OFFSET, float(elem.period))
        emb = backend.at_add(emb, 8 + POSITION_OFFSET + 1, float(elem.group) / 10.0)
        emb = backend.at_add(emb, 8 + POSITION_OFFSET + 2, BLOCK_CODES.get(elem.block, 0.0))
        emb = backend.at_add(emb, 8 + POSITION_OFFSET + 3, float(elem.valence_electrons) / 8.0)

        # Electronegativity
        if elem.electronegativity is not None:
            emb = backend.at_add(emb, 8 + ELECTRONEGATIVITY_OFFSET, 1.0)
            emb = backend.at_add(emb, 8 + ELECTRONEGATIVITY_OFFSET + 1, 
                elem.electronegativity / 4.0  # Normalize to ~[0,1]
            )

        # Oxidation states (up to 4)
        for i, ox in enumerate(elem.common_oxidation_states[:4]):
            emb = backend.at_add(emb, 8 + OXIDATION_OFFSET + i, ox / 8.0)  # Normalize

        # Category
        cat_code = CATEGORY_CODES.get(elem.category, 0)
        emb = backend.at_add(emb, 8 + CATEGORY_OFFSET, float(cat_code) / 10.0)

        return emb

    def decode(self, emb: Any) -> Element:
        """
        Decode embedding to element (by nearest atomic number).

        Args:
            emb: 128-dim embedding

        Returns:
            Element object
        """
        backend = get_backend()
        # Extract atomic number
        log_z = emb[8 + ATOMIC_NUM_OFFSET + 1].item()
        z = int(round(backend.exp(backend.array(log_z)).item()))
        z = max(1, min(118, z))  # Clamp to valid range

        elem = PERIODIC_TABLE.get(z)
        if elem is None:
            # Fallback to hydrogen
            elem = PERIODIC_TABLE.get(1)

        return elem

    def decode_symbol(self, emb: Any) -> str:
        """Decode embedding to element symbol."""
        return self.decode(emb).symbol

    def decode_atomic_number(self, emb: Any) -> int:
        """Decode embedding to atomic number."""
        return self.decode(emb).atomic_number

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid element."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Chemical Property Queries
    # =========================================================================

    def electronegativity_difference(self, emb1: Any, emb2: Any) -> float:
        """
        Calculate electronegativity difference between two elements.

        Useful for predicting bond polarity.
        """
        has_en1 = emb1[8 + ELECTRONEGATIVITY_OFFSET].item() > 0.5
        has_en2 = emb2[8 + ELECTRONEGATIVITY_OFFSET].item() > 0.5

        if not (has_en1 and has_en2):
            return 0.0

        en1 = emb1[8 + ELECTRONEGATIVITY_OFFSET + 1].item() * 4.0
        en2 = emb2[8 + ELECTRONEGATIVITY_OFFSET + 1].item() * 4.0

        return abs(en1 - en2)

    def same_group(self, emb1: Any, emb2: Any) -> bool:
        """Check if two elements are in the same group."""
        g1 = emb1[8 + POSITION_OFFSET + 1].item() * 10
        g2 = emb2[8 + POSITION_OFFSET + 1].item() * 10
        return abs(g1 - g2) < 0.5

    def same_period(self, emb1: Any, emb2: Any) -> bool:
        """Check if two elements are in the same period."""
        p1 = emb1[8 + POSITION_OFFSET].item()
        p2 = emb2[8 + POSITION_OFFSET].item()
        return abs(p1 - p2) < 0.5

    def is_metal(self, emb: Any) -> bool:
        """Check if element is a metal."""
        cat = int(emb[8 + CATEGORY_OFFSET].item() * 10)
        # Categories 0-3 are metals, 8-9 are lanthanides/actinides (also metals)
        return cat in [0, 1, 2, 3, 8, 9]

    def is_nonmetal(self, emb: Any) -> bool:
        """Check if element is a nonmetal."""
        cat = int(emb[8 + CATEGORY_OFFSET].item() * 10)
        return cat in [5, 6, 7]  # nonmetal, halogen, noble gas


def get_element(symbol_or_z: Union[str, int]) -> Optional[Element]:
    """Convenience function to get element data."""
    return PERIODIC_TABLE.get(symbol_or_z)
