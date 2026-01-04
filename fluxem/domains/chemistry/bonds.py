"""
Chemical Bond Encoder.

Embeds chemical bonds with their properties: bond type, order, polarity,
participating elements, and structural information.

Bond operations like combination and comparison are supported.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


class BondType(Enum):
    """Types of chemical bonds."""
    COVALENT = 0       # Shared electrons
    IONIC = 1          # Electron transfer
    METALLIC = 2       # Electron sea
    HYDROGEN = 3       # H-bond (weak)
    VAN_DER_WAALS = 4  # Dispersion forces (weakest)
    COORDINATE = 5     # Dative/coordinate covalent


class BondOrder(Enum):
    """Bond order (multiplicity)."""
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 1.5     # Resonance average
    PARTIAL = 0.5      # Hydrogen bonds, etc.


@dataclass
class Bond:
    """Represents a chemical bond between two atoms."""
    element1: str           # First element symbol
    element2: str           # Second element symbol
    bond_type: BondType
    bond_order: Union[int, float]
    length: Optional[float] = None     # Bond length in Angstroms
    energy: Optional[float] = None     # Bond energy in kJ/mol
    polarity: Optional[float] = None   # Electronegativity difference
    is_polar: bool = False
    is_rotatable: bool = True          # Can rotate around this bond


# =============================================================================
# Bond Data (common bonds)
# =============================================================================

# Common bond lengths in Angstroms
BOND_LENGTHS: Dict[Tuple[str, str, int], float] = {
    # Single bonds
    ("C", "C", 1): 1.54,
    ("C", "H", 1): 1.09,
    ("C", "O", 1): 1.43,
    ("C", "N", 1): 1.47,
    ("C", "S", 1): 1.82,
    ("C", "Cl", 1): 1.77,
    ("C", "Br", 1): 1.94,
    ("C", "F", 1): 1.35,
    ("O", "H", 1): 0.96,
    ("N", "H", 1): 1.01,
    ("S", "H", 1): 1.34,
    ("O", "O", 1): 1.48,
    ("N", "N", 1): 1.45,
    # Double bonds
    ("C", "C", 2): 1.34,
    ("C", "O", 2): 1.23,
    ("C", "N", 2): 1.29,
    ("O", "O", 2): 1.21,
    ("N", "N", 2): 1.25,
    ("C", "S", 2): 1.60,
    # Triple bonds
    ("C", "C", 3): 1.20,
    ("C", "N", 3): 1.16,
    ("N", "N", 3): 1.10,
}

# Common bond energies in kJ/mol
BOND_ENERGIES: Dict[Tuple[str, str, int], float] = {
    # Single bonds
    ("C", "C", 1): 347,
    ("C", "H", 1): 413,
    ("C", "O", 1): 358,
    ("C", "N", 1): 305,
    ("C", "S", 1): 272,
    ("C", "Cl", 1): 339,
    ("C", "Br", 1): 276,
    ("C", "F", 1): 485,
    ("O", "H", 1): 463,
    ("N", "H", 1): 391,
    ("S", "H", 1): 363,
    ("O", "O", 1): 146,
    ("N", "N", 1): 163,
    # Double bonds
    ("C", "C", 2): 614,
    ("C", "O", 2): 745,
    ("C", "N", 2): 615,
    ("O", "O", 2): 498,
    ("N", "N", 2): 418,
    # Triple bonds
    ("C", "C", 3): 839,
    ("C", "N", 3): 891,
    ("N", "N", 3): 945,
}

# Electronegativity values (Pauling scale)
ELECTRONEGATIVITIES: Dict[str, float] = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "S": 2.58, "P": 2.19, "Cl": 3.16, "Br": 2.96, "I": 2.66,
    "Si": 1.90, "B": 2.04, "Na": 0.93, "K": 0.82, "Ca": 1.00,
    "Mg": 1.31, "Al": 1.61, "Fe": 1.83, "Cu": 1.90, "Zn": 1.65,
}


# =============================================================================
# Embedding Layout
# =============================================================================

# Layout within domain-specific region (dims 8-71):
# dims 0-1:   Element 1 atomic number (sign, log|Z|)
# dims 2-3:   Element 2 atomic number (sign, log|Z|)
# dims 4:     Bond type (0-5, normalized)
# dims 5:     Bond order (0.5-3, normalized)
# dims 6-7:   Bond length (sign, log|length|)
# dims 8-9:   Bond energy (sign, log|energy|)
# dims 10:    Polarity (electronegativity difference / 4)
# dims 11:    Is polar flag
# dims 12:    Is rotatable flag
# dims 13-15: Reserved flags
# dims 16-31: Element property encoding (hash of pair)

ELEM1_OFFSET = 0
ELEM2_OFFSET = 2
BOND_TYPE_OFFSET = 4
BOND_ORDER_OFFSET = 5
LENGTH_OFFSET = 6
ENERGY_OFFSET = 8
POLARITY_OFFSET = 10
IS_POLAR_FLAG = 11
IS_ROTATABLE_FLAG = 12


# Element symbol to atomic number mapping
ELEMENT_TO_Z: Dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Fe": 26, "Cu": 29,
    "Zn": 30, "Br": 35, "I": 53, "Au": 79, "Hg": 80, "Pb": 82,
}

Z_TO_ELEMENT: Dict[int, str] = {v: k for k, v in ELEMENT_TO_Z.items()}


def _get_bond_length(e1: str, e2: str, order: int) -> Optional[float]:
    """Look up bond length, trying both orderings."""
    key = (e1, e2, order)
    if key in BOND_LENGTHS:
        return BOND_LENGTHS[key]
    key = (e2, e1, order)
    if key in BOND_LENGTHS:
        return BOND_LENGTHS[key]
    return None


def _get_bond_energy(e1: str, e2: str, order: int) -> Optional[float]:
    """Look up bond energy, trying both orderings."""
    key = (e1, e2, order)
    if key in BOND_ENERGIES:
        return BOND_ENERGIES[key]
    key = (e2, e1, order)
    if key in BOND_ENERGIES:
        return BOND_ENERGIES[key]
    return None


class BondEncoder:
    """
    Encoder for chemical bonds.

    Encodes bond properties including type, order, length, energy, and polarity.
    """

    domain_tag = DOMAIN_TAGS["chem_bond"]
    domain_name = "chem_bond"

    def encode(self, bond: Union[Bond, Tuple[str, str, int]]) -> Any:
        """
        Encode a chemical bond.

        Args:
            bond: Bond object or tuple of (element1, element2, bond_order)

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(bond, tuple):
            e1, e2, order = bond
            bond = self._create_bond(e1, e2, order)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Element atomic numbers
        z1 = ELEMENT_TO_Z.get(bond.element1, 6)  # Default to carbon
        z2 = ELEMENT_TO_Z.get(bond.element2, 6)

        sign1, log1 = log_encode_value(float(z1))
        emb = backend.at_add(emb, 8 + ELEM1_OFFSET, sign1)
        emb = backend.at_add(emb, 8 + ELEM1_OFFSET + 1, log1)

        sign2, log2 = log_encode_value(float(z2))
        emb = backend.at_add(emb, 8 + ELEM2_OFFSET, sign2)
        emb = backend.at_add(emb, 8 + ELEM2_OFFSET + 1, log2)

        # Bond type (normalized to [0, 1])
        emb = backend.at_add(emb, 8 + BOND_TYPE_OFFSET, bond.bond_type.value / 5.0)

        # Bond order (normalized, 0.5-3 -> 0-1)
        order_norm = (bond.bond_order - 0.5) / 2.5
        emb = backend.at_add(emb, 8 + BOND_ORDER_OFFSET, order_norm)

        # Bond length
        if bond.length is not None:
            sign_l, log_l = log_encode_value(bond.length)
            emb = backend.at_add(emb, 8 + LENGTH_OFFSET, sign_l)
            emb = backend.at_add(emb, 8 + LENGTH_OFFSET + 1, log_l)

        # Bond energy
        if bond.energy is not None:
            sign_e, log_e = log_encode_value(bond.energy)
            emb = backend.at_add(emb, 8 + ENERGY_OFFSET, sign_e)
            emb = backend.at_add(emb, 8 + ENERGY_OFFSET + 1, log_e)

        # Polarity
        if bond.polarity is not None:
            emb = backend.at_add(emb, 8 + POLARITY_OFFSET, bond.polarity / 4.0)

        # Flags
        emb = backend.at_add(emb, 8 + IS_POLAR_FLAG, 1.0 if bond.is_polar else 0.0)
        emb = backend.at_add(emb, 8 + IS_ROTATABLE_FLAG, 1.0 if bond.is_rotatable else 0.0)

        return emb

    def _create_bond(self, e1: str, e2: str, order: int) -> Bond:
        """Create a Bond object from element symbols and order."""
        # Determine bond type based on electronegativity
        en1 = ELECTRONEGATIVITIES.get(e1, 2.0)
        en2 = ELECTRONEGATIVITIES.get(e2, 2.0)
        polarity = abs(en1 - en2)

        # Classify bond type
        if polarity > 1.7:
            bond_type = BondType.IONIC
        else:
            bond_type = BondType.COVALENT

        is_polar = polarity > 0.4

        # Look up properties
        length = _get_bond_length(e1, e2, order)
        energy = _get_bond_energy(e1, e2, order)

        # Rotatable? Double and triple bonds are not
        is_rotatable = (order == 1) and bond_type == BondType.COVALENT

        return Bond(
            element1=e1,
            element2=e2,
            bond_type=bond_type,
            bond_order=order,
            length=length,
            energy=energy,
            polarity=polarity,
            is_polar=is_polar,
            is_rotatable=is_rotatable,
        )

    def decode(self, emb: Any) -> Bond:
        """
        Decode embedding to Bond object.

        Args:
            emb: 128-dim embedding

        Returns:
            Bond object
        """
        # Decode element atomic numbers
        sign1 = emb[8 + ELEM1_OFFSET].item()
        log1 = emb[8 + ELEM1_OFFSET + 1].item()
        z1 = int(round(log_decode_value(sign1, log1)))
        z1 = max(1, min(118, z1))

        sign2 = emb[8 + ELEM2_OFFSET].item()
        log2 = emb[8 + ELEM2_OFFSET + 1].item()
        z2 = int(round(log_decode_value(sign2, log2)))
        z2 = max(1, min(118, z2))

        e1 = Z_TO_ELEMENT.get(z1, "C")
        e2 = Z_TO_ELEMENT.get(z2, "C")

        # Decode bond type
        type_val = int(round(emb[8 + BOND_TYPE_OFFSET].item() * 5.0))
        type_val = max(0, min(5, type_val))
        bond_type = BondType(type_val)

        # Decode bond order
        order_norm = emb[8 + BOND_ORDER_OFFSET].item()
        bond_order = order_norm * 2.5 + 0.5
        # Round to nearest valid order
        if bond_order < 0.75:
            bond_order = 0.5
        elif bond_order < 1.25:
            bond_order = 1
        elif bond_order < 1.75:
            bond_order = 1.5
        elif bond_order < 2.5:
            bond_order = 2
        else:
            bond_order = 3

        # Decode length
        sign_l = emb[8 + LENGTH_OFFSET].item()
        log_l = emb[8 + LENGTH_OFFSET + 1].item()
        length = log_decode_value(sign_l, log_l) if abs(sign_l) > 0.5 else None

        # Decode energy
        sign_e = emb[8 + ENERGY_OFFSET].item()
        log_e = emb[8 + ENERGY_OFFSET + 1].item()
        energy = log_decode_value(sign_e, log_e) if abs(sign_e) > 0.5 else None

        # Decode polarity
        polarity = emb[8 + POLARITY_OFFSET].item() * 4.0

        # Decode flags
        is_polar = emb[8 + IS_POLAR_FLAG].item() > 0.5
        is_rotatable = emb[8 + IS_ROTATABLE_FLAG].item() > 0.5

        return Bond(
            element1=e1,
            element2=e2,
            bond_type=bond_type,
            bond_order=bond_order,
            length=length,
            energy=energy,
            polarity=polarity,
            is_polar=is_polar,
            is_rotatable=is_rotatable,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid bond."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Bond Property Queries
    # =========================================================================

    def get_elements(self, emb: Any) -> Tuple[str, str]:
        """Get the elements involved in the bond."""
        bond = self.decode(emb)
        return (bond.element1, bond.element2)

    def get_bond_order(self, emb: Any) -> float:
        """Get the bond order."""
        order_norm = emb[8 + BOND_ORDER_OFFSET].item()
        return order_norm * 2.5 + 0.5

    def get_bond_type(self, emb: Any) -> BondType:
        """Get the bond type."""
        type_val = int(round(emb[8 + BOND_TYPE_OFFSET].item() * 5.0))
        type_val = max(0, min(5, type_val))
        return BondType(type_val)

    def get_length(self, emb: Any) -> Optional[float]:
        """Get bond length in Angstroms."""
        sign = emb[8 + LENGTH_OFFSET].item()
        if abs(sign) < 0.5:
            return None
        log_l = emb[8 + LENGTH_OFFSET + 1].item()
        return log_decode_value(sign, log_l)

    def get_energy(self, emb: Any) -> Optional[float]:
        """Get bond energy in kJ/mol."""
        sign = emb[8 + ENERGY_OFFSET].item()
        if abs(sign) < 0.5:
            return None
        log_e = emb[8 + ENERGY_OFFSET + 1].item()
        return log_decode_value(sign, log_e)

    def get_polarity(self, emb: Any) -> float:
        """Get bond polarity (electronegativity difference)."""
        return emb[8 + POLARITY_OFFSET].item() * 4.0

    def is_polar(self, emb: Any) -> bool:
        """Check if bond is polar."""
        return emb[8 + IS_POLAR_FLAG].item() > 0.5

    def is_rotatable(self, emb: Any) -> bool:
        """Check if bond is rotatable."""
        return emb[8 + IS_ROTATABLE_FLAG].item() > 0.5

    def is_single(self, emb: Any) -> bool:
        """Check if this is a single bond."""
        order = self.get_bond_order(emb)
        return abs(order - 1.0) < 0.25

    def is_double(self, emb: Any) -> bool:
        """Check if this is a double bond."""
        order = self.get_bond_order(emb)
        return abs(order - 2.0) < 0.25

    def is_triple(self, emb: Any) -> bool:
        """Check if this is a triple bond."""
        order = self.get_bond_order(emb)
        return abs(order - 3.0) < 0.25

    def is_covalent(self, emb: Any) -> bool:
        """Check if this is a covalent bond."""
        return self.get_bond_type(emb) == BondType.COVALENT

    def is_ionic(self, emb: Any) -> bool:
        """Check if this is an ionic bond."""
        return self.get_bond_type(emb) == BondType.IONIC

    # =========================================================================
    # Bond Comparisons
    # =========================================================================

    def compare_strength(self, emb1: Any, emb2: Any) -> int:
        """
        Compare bond strengths.

        Returns:
            -1 if bond1 < bond2, 0 if equal, 1 if bond1 > bond2
        """
        e1 = self.get_energy(emb1)
        e2 = self.get_energy(emb2)

        if e1 is None or e2 is None:
            # Fall back to bond order comparison
            o1 = self.get_bond_order(emb1)
            o2 = self.get_bond_order(emb2)
            if abs(o1 - o2) < 0.1:
                return 0
            return 1 if o1 > o2 else -1

        if abs(e1 - e2) < 1.0:  # Within 1 kJ/mol
            return 0
        return 1 if e1 > e2 else -1

    def same_elements(self, emb1: Any, emb2: Any) -> bool:
        """Check if two bonds involve the same elements (order-independent)."""
        e1a, e1b = self.get_elements(emb1)
        e2a, e2b = self.get_elements(emb2)

        return (e1a == e2a and e1b == e2b) or (e1a == e2b and e1b == e2a)


# =============================================================================
# Convenience Functions
# =============================================================================

def encode_bond(element1: str, element2: str, order: int = 1) -> Any:
    """Encode a bond from element symbols and order."""
    encoder = BondEncoder()
    return encoder.encode((element1, element2, order))


def decode_bond(emb: Any) -> Bond:
    """Decode a bond embedding."""
    encoder = BondEncoder()
    return encoder.decode(emb)


def get_bond_info(element1: str, element2: str, order: int = 1) -> Dict:
    """Get bond information as a dictionary."""
    encoder = BondEncoder()
    bond = encoder._create_bond(element1, element2, order)
    return {
        "elements": (bond.element1, bond.element2),
        "type": bond.bond_type.name,
        "order": bond.bond_order,
        "length": bond.length,
        "energy": bond.energy,
        "polarity": bond.polarity,
        "is_polar": bond.is_polar,
        "is_rotatable": bond.is_rotatable,
    }
