"""
Unit Encoder for Physics.

Embeds SI units and derived units with their dimensions and conversion factors.
Supports:
- SI base units (m, kg, s, A, K, mol, cd)
- Derived SI units (N, J, W, Pa, Hz, C, V, Ohm, etc.)
- Unit prefixes (kilo, milli, micro, nano, etc.)
- Unit parsing (e.g., "m/s", "kg*m/s^2", "kN")
- Conversion between compatible units

Conversions are deterministic given the encoded fields.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import re
import math
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)
from .dimensions import Dimensions, SI_DIMENSIONS


# =============================================================================
# Unit Prefixes (SI)
# =============================================================================

# Prefix name -> (symbol, power of 10)
SI_PREFIXES: Dict[str, Tuple[str, int]] = {
    "yotta": ("Y", 24),
    "zetta": ("Z", 21),
    "exa": ("E", 18),
    "peta": ("P", 15),
    "tera": ("T", 12),
    "giga": ("G", 9),
    "mega": ("M", 6),
    "kilo": ("k", 3),
    "hecto": ("h", 2),
    "deca": ("da", 1),
    "deci": ("d", -1),
    "centi": ("c", -2),
    "milli": ("m", -3),
    "micro": ("u", -6),  # Also handle mu symbol
    "nano": ("n", -9),
    "pico": ("p", -12),
    "femto": ("f", -15),
    "atto": ("a", -18),
    "zepto": ("z", -21),
    "yocto": ("y", -24),
}

# Symbol -> power of 10
PREFIX_SYMBOLS: Dict[str, int] = {
    "Y": 24, "Z": 21, "E": 18, "P": 15, "T": 12, "G": 9, "M": 6,
    "k": 3, "h": 2, "da": 1,
    "d": -1, "c": -2, "m": -3, "u": -6, "\u03bc": -6,  # mu symbol
    "n": -9, "p": -12, "f": -15, "a": -18, "z": -21, "y": -24,
}


# =============================================================================
# SI Base Units
# =============================================================================

@dataclass(frozen=True)
class UnitDefinition:
    """Definition of a unit with its dimensions and SI conversion factor."""
    symbol: str
    name: str
    dimensions: Dimensions
    to_si_factor: float  # Multiply by this to get SI base units
    is_base: bool = False


# SI Base Units - conversion factor is 1.0
SI_BASE_UNITS: Dict[str, UnitDefinition] = {
    "m": UnitDefinition("m", "meter", Dimensions(L=1), 1.0, True),
    "kg": UnitDefinition("kg", "kilogram", Dimensions(M=1), 1.0, True),
    "s": UnitDefinition("s", "second", Dimensions(T=1), 1.0, True),
    "A": UnitDefinition("A", "ampere", Dimensions(I=1), 1.0, True),
    "K": UnitDefinition("K", "kelvin", Dimensions(Theta=1), 1.0, True),
    "mol": UnitDefinition("mol", "mole", Dimensions(N=1), 1.0, True),
    "cd": UnitDefinition("cd", "candela", Dimensions(J=1), 1.0, True),
}

# Alternative symbols for base units
BASE_UNIT_ALIASES: Dict[str, str] = {
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "kilogram": "kg", "kilograms": "kg",
    "second": "s", "seconds": "s", "sec": "s",
    "ampere": "A", "amperes": "A", "amp": "A", "amps": "A",
    "kelvin": "K",
    "mole": "mol", "moles": "mol",
    "candela": "cd",
}


# =============================================================================
# Derived SI Units
# =============================================================================

DERIVED_UNITS: Dict[str, UnitDefinition] = {
    # Mechanical
    "N": UnitDefinition("N", "newton", Dimensions(M=1, L=1, T=-2), 1.0),
    "J": UnitDefinition("J", "joule", Dimensions(M=1, L=2, T=-2), 1.0),
    "W": UnitDefinition("W", "watt", Dimensions(M=1, L=2, T=-3), 1.0),
    "Pa": UnitDefinition("Pa", "pascal", Dimensions(M=1, L=-1, T=-2), 1.0),
    "Hz": UnitDefinition("Hz", "hertz", Dimensions(T=-1), 1.0),

    # Electromagnetic
    "C": UnitDefinition("C", "coulomb", Dimensions(I=1, T=1), 1.0),
    "V": UnitDefinition("V", "volt", Dimensions(M=1, L=2, T=-3, I=-1), 1.0),
    "\u03a9": UnitDefinition("\u03a9", "ohm", Dimensions(M=1, L=2, T=-3, I=-2), 1.0),
    "Ohm": UnitDefinition("Ohm", "ohm", Dimensions(M=1, L=2, T=-3, I=-2), 1.0),
    "ohm": UnitDefinition("ohm", "ohm", Dimensions(M=1, L=2, T=-3, I=-2), 1.0),
    "S": UnitDefinition("S", "siemens", Dimensions(M=-1, L=-2, T=3, I=2), 1.0),
    "F": UnitDefinition("F", "farad", Dimensions(M=-1, L=-2, T=4, I=2), 1.0),
    "H": UnitDefinition("H", "henry", Dimensions(M=1, L=2, T=-2, I=-2), 1.0),
    "Wb": UnitDefinition("Wb", "weber", Dimensions(M=1, L=2, T=-2, I=-1), 1.0),
    "T": UnitDefinition("T", "tesla", Dimensions(M=1, T=-2, I=-1), 1.0),

    # Other
    "lm": UnitDefinition("lm", "lumen", Dimensions(J=1), 1.0),
    "lx": UnitDefinition("lx", "lux", Dimensions(L=-2, J=1), 1.0),
    "Bq": UnitDefinition("Bq", "becquerel", Dimensions(T=-1), 1.0),
    "Gy": UnitDefinition("Gy", "gray", Dimensions(L=2, T=-2), 1.0),
    "Sv": UnitDefinition("Sv", "sievert", Dimensions(L=2, T=-2), 1.0),
    "kat": UnitDefinition("kat", "katal", Dimensions(T=-1, N=1), 1.0),
    "rad": UnitDefinition("rad", "radian", Dimensions(), 1.0),  # dimensionless
    "sr": UnitDefinition("sr", "steradian", Dimensions(), 1.0),  # dimensionless
}

# Common non-SI units with conversion to SI
NON_SI_UNITS: Dict[str, UnitDefinition] = {
    # Length
    "cm": UnitDefinition("cm", "centimeter", Dimensions(L=1), 1e-2),
    "mm": UnitDefinition("mm", "millimeter", Dimensions(L=1), 1e-3),
    "km": UnitDefinition("km", "kilometer", Dimensions(L=1), 1e3),
    "um": UnitDefinition("um", "micrometer", Dimensions(L=1), 1e-6),
    "\u03bcm": UnitDefinition("\u03bcm", "micrometer", Dimensions(L=1), 1e-6),
    "nm": UnitDefinition("nm", "nanometer", Dimensions(L=1), 1e-9),
    "in": UnitDefinition("in", "inch", Dimensions(L=1), 0.0254),
    "ft": UnitDefinition("ft", "foot", Dimensions(L=1), 0.3048),
    "mi": UnitDefinition("mi", "mile", Dimensions(L=1), 1609.344),
    "yd": UnitDefinition("yd", "yard", Dimensions(L=1), 0.9144),

    # Mass
    "g": UnitDefinition("g", "gram", Dimensions(M=1), 1e-3),
    "mg": UnitDefinition("mg", "milligram", Dimensions(M=1), 1e-6),
    "ug": UnitDefinition("ug", "microgram", Dimensions(M=1), 1e-9),
    "t": UnitDefinition("t", "tonne", Dimensions(M=1), 1e3),
    "lb": UnitDefinition("lb", "pound", Dimensions(M=1), 0.45359237),
    "oz": UnitDefinition("oz", "ounce", Dimensions(M=1), 0.028349523125),

    # Time
    "ms": UnitDefinition("ms", "millisecond", Dimensions(T=1), 1e-3),
    "us": UnitDefinition("us", "microsecond", Dimensions(T=1), 1e-6),
    "ns": UnitDefinition("ns", "nanosecond", Dimensions(T=1), 1e-9),
    "min": UnitDefinition("min", "minute", Dimensions(T=1), 60.0),
    "h": UnitDefinition("h", "hour", Dimensions(T=1), 3600.0),
    "hr": UnitDefinition("hr", "hour", Dimensions(T=1), 3600.0),
    "d": UnitDefinition("d", "day", Dimensions(T=1), 86400.0),
    "day": UnitDefinition("day", "day", Dimensions(T=1), 86400.0),

    # Force
    "kN": UnitDefinition("kN", "kilonewton", Dimensions(M=1, L=1, T=-2), 1e3),
    "MN": UnitDefinition("MN", "meganewton", Dimensions(M=1, L=1, T=-2), 1e6),
    "mN": UnitDefinition("mN", "millinewton", Dimensions(M=1, L=1, T=-2), 1e-3),
    "dyn": UnitDefinition("dyn", "dyne", Dimensions(M=1, L=1, T=-2), 1e-5),
    "lbf": UnitDefinition("lbf", "pound-force", Dimensions(M=1, L=1, T=-2), 4.4482216152605),

    # Energy
    "kJ": UnitDefinition("kJ", "kilojoule", Dimensions(M=1, L=2, T=-2), 1e3),
    "MJ": UnitDefinition("MJ", "megajoule", Dimensions(M=1, L=2, T=-2), 1e6),
    "mJ": UnitDefinition("mJ", "millijoule", Dimensions(M=1, L=2, T=-2), 1e-3),
    "eV": UnitDefinition("eV", "electronvolt", Dimensions(M=1, L=2, T=-2), 1.602176634e-19),
    "keV": UnitDefinition("keV", "kiloelectronvolt", Dimensions(M=1, L=2, T=-2), 1.602176634e-16),
    "MeV": UnitDefinition("MeV", "megaelectronvolt", Dimensions(M=1, L=2, T=-2), 1.602176634e-13),
    "cal": UnitDefinition("cal", "calorie", Dimensions(M=1, L=2, T=-2), 4.184),
    "kcal": UnitDefinition("kcal", "kilocalorie", Dimensions(M=1, L=2, T=-2), 4184.0),
    "Wh": UnitDefinition("Wh", "watt-hour", Dimensions(M=1, L=2, T=-2), 3600.0),
    "kWh": UnitDefinition("kWh", "kilowatt-hour", Dimensions(M=1, L=2, T=-2), 3.6e6),

    # Power
    "kW": UnitDefinition("kW", "kilowatt", Dimensions(M=1, L=2, T=-3), 1e3),
    "MW": UnitDefinition("MW", "megawatt", Dimensions(M=1, L=2, T=-3), 1e6),
    "GW": UnitDefinition("GW", "gigawatt", Dimensions(M=1, L=2, T=-3), 1e9),
    "mW": UnitDefinition("mW", "milliwatt", Dimensions(M=1, L=2, T=-3), 1e-3),
    "hp": UnitDefinition("hp", "horsepower", Dimensions(M=1, L=2, T=-3), 745.7),

    # Pressure
    "kPa": UnitDefinition("kPa", "kilopascal", Dimensions(M=1, L=-1, T=-2), 1e3),
    "MPa": UnitDefinition("MPa", "megapascal", Dimensions(M=1, L=-1, T=-2), 1e6),
    "GPa": UnitDefinition("GPa", "gigapascal", Dimensions(M=1, L=-1, T=-2), 1e9),
    "bar": UnitDefinition("bar", "bar", Dimensions(M=1, L=-1, T=-2), 1e5),
    "mbar": UnitDefinition("mbar", "millibar", Dimensions(M=1, L=-1, T=-2), 1e2),
    "atm": UnitDefinition("atm", "atmosphere", Dimensions(M=1, L=-1, T=-2), 101325.0),
    "psi": UnitDefinition("psi", "pound per square inch", Dimensions(M=1, L=-1, T=-2), 6894.757293168),
    "torr": UnitDefinition("torr", "torr", Dimensions(M=1, L=-1, T=-2), 133.32236842105),
    "mmHg": UnitDefinition("mmHg", "millimeter of mercury", Dimensions(M=1, L=-1, T=-2), 133.32236842105),

    # Frequency
    "kHz": UnitDefinition("kHz", "kilohertz", Dimensions(T=-1), 1e3),
    "MHz": UnitDefinition("MHz", "megahertz", Dimensions(T=-1), 1e6),
    "GHz": UnitDefinition("GHz", "gigahertz", Dimensions(T=-1), 1e9),
    "rpm": UnitDefinition("rpm", "revolutions per minute", Dimensions(T=-1), 1.0/60.0),

    # Electrical
    "mA": UnitDefinition("mA", "milliampere", Dimensions(I=1), 1e-3),
    "uA": UnitDefinition("uA", "microampere", Dimensions(I=1), 1e-6),
    "nA": UnitDefinition("nA", "nanoampere", Dimensions(I=1), 1e-9),
    "kV": UnitDefinition("kV", "kilovolt", Dimensions(M=1, L=2, T=-3, I=-1), 1e3),
    "mV": UnitDefinition("mV", "millivolt", Dimensions(M=1, L=2, T=-3, I=-1), 1e-3),
    "uV": UnitDefinition("uV", "microvolt", Dimensions(M=1, L=2, T=-3, I=-1), 1e-6),
    "pF": UnitDefinition("pF", "picofarad", Dimensions(M=-1, L=-2, T=4, I=2), 1e-12),
    "nF": UnitDefinition("nF", "nanofarad", Dimensions(M=-1, L=-2, T=4, I=2), 1e-9),
    "uF": UnitDefinition("uF", "microfarad", Dimensions(M=-1, L=-2, T=4, I=2), 1e-6),
    "mF": UnitDefinition("mF", "millifarad", Dimensions(M=-1, L=-2, T=4, I=2), 1e-3),

    # Angle (dimensionless but with conversion)
    "deg": UnitDefinition("deg", "degree", Dimensions(), math.pi/180.0),
    "\u00b0": UnitDefinition("\u00b0", "degree", Dimensions(), math.pi/180.0),

    # Area
    "ha": UnitDefinition("ha", "hectare", Dimensions(L=2), 1e4),
    "acre": UnitDefinition("acre", "acre", Dimensions(L=2), 4046.8564224),

    # Volume
    "L": UnitDefinition("L", "liter", Dimensions(L=3), 1e-3),
    "mL": UnitDefinition("mL", "milliliter", Dimensions(L=3), 1e-6),
    "gal": UnitDefinition("gal", "gallon", Dimensions(L=3), 3.785411784e-3),
}

# Build complete unit dictionary
ALL_UNITS: Dict[str, UnitDefinition] = {}
ALL_UNITS.update(SI_BASE_UNITS)
ALL_UNITS.update(DERIVED_UNITS)
ALL_UNITS.update(NON_SI_UNITS)


# =============================================================================
# Embedding Layout (within domain-specific region, 64 dims at offset 8)
# =============================================================================

# dims 0-6:   SI exponents [M, L, T, I, Theta, N, J]
# dim 7:      Dimensionless flag
# dims 8-9:   Conversion factor to SI (sign, log|factor|)
# dim 10:     Is SI base unit flag
# dim 11:     Is derived SI unit flag
# dims 12-19: Unit identifier hash (8 dims)
# dims 20-27: Prefix encoding (8 dims for prefix power)
# dims 28-63: Reserved

UNIT_DIM_OFFSET = 0
UNIT_DIM_SIZE = 7
UNIT_DIMENSIONLESS_FLAG = 7
UNIT_CONV_SIGN = 8
UNIT_CONV_LOG = 9
UNIT_IS_SI_BASE = 10
UNIT_IS_SI_DERIVED = 11
UNIT_ID_HASH_OFFSET = 12
UNIT_ID_HASH_SIZE = 8
UNIT_PREFIX_OFFSET = 20
UNIT_PREFIX_SIZE = 8


# =============================================================================
# Unit Parser
# =============================================================================

@dataclass
class ParsedUnit:
    """Result of parsing a unit string."""
    base_units: List[Tuple[str, int]]  # List of (unit_symbol, exponent)
    total_factor: float  # Combined conversion factor
    dimensions: Dimensions  # Combined dimensions
    original: str  # Original string


def _simple_hash(s: str, size: int = 8) -> List[float]:
    """Create a simple hash vector for unit identification."""
    h = hash(s)
    result = []
    for i in range(size):
        # Extract bits and normalize to [-1, 1]
        bit = (h >> (i * 4)) & 0xF
        result.append((bit - 7.5) / 7.5)
    return result


def _tokenize_unit_string(unit_str: str) -> List[str]:
    """Tokenize a unit string into unit tokens."""
    # Normalize common separators
    s = unit_str.strip()
    s = s.replace("\u22c5", "*")  # cdot
    s = s.replace("\u00b7", "*")  # middle dot
    s = s.replace("\u2022", "*")  # bullet
    s = s.replace(" ", "*")

    # Handle superscript exponents
    superscripts = {
        "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
        "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
        "\u2078": "8", "\u2079": "9", "\u207b": "-"
    }
    for sup, normal in superscripts.items():
        s = s.replace(sup, "^" + normal)

    # Clean up consecutive caret signs
    s = re.sub(r"\^+", "^", s)

    return s


def parse_unit(unit_str: str) -> ParsedUnit:
    """
    Parse a unit string like "m/s", "kg*m/s^2", "kN".

    Returns:
        ParsedUnit with components, factor, and dimensions.
    """
    s = _tokenize_unit_string(unit_str)

    # Split by multiplication and division
    # Handle division by splitting on /
    parts = s.split("/")
    numerator = parts[0] if parts else ""
    denominator = "*".join(parts[1:]) if len(parts) > 1 else ""

    base_units: List[Tuple[str, int]] = []
    total_factor = 1.0
    total_dims = Dimensions()

    def process_term(term: str, sign: int):
        """Process a single unit term, updating accumulators."""
        nonlocal total_factor, total_dims

        if not term:
            return

        # Parse exponent
        if "^" in term:
            unit_part, exp_part = term.split("^", 1)
            try:
                exponent = int(exp_part) * sign
            except ValueError:
                exponent = sign
        else:
            unit_part = term
            exponent = sign

        # Look up unit
        unit_def = _lookup_unit(unit_part)
        if unit_def is None:
            # Try to parse as prefixed unit
            unit_def, prefix_factor = _parse_prefixed_unit(unit_part)
            if unit_def is None:
                raise ValueError(f"Unknown unit: {unit_part}")
            total_factor *= prefix_factor ** exponent

        base_units.append((unit_def.symbol, exponent))
        total_factor *= unit_def.to_si_factor ** exponent

        # Update dimensions
        if exponent > 0:
            for _ in range(exponent):
                total_dims = total_dims * unit_def.dimensions
        else:
            for _ in range(-exponent):
                total_dims = total_dims / unit_def.dimensions

    # Process numerator
    for term in numerator.split("*"):
        term = term.strip()
        if term:
            process_term(term, 1)

    # Process denominator
    for term in denominator.split("*"):
        term = term.strip()
        if term:
            process_term(term, -1)

    return ParsedUnit(
        base_units=base_units,
        total_factor=total_factor,
        dimensions=total_dims,
        original=unit_str
    )


def _lookup_unit(symbol: str) -> Optional[UnitDefinition]:
    """Look up a unit by symbol or alias."""
    if symbol in ALL_UNITS:
        return ALL_UNITS[symbol]
    if symbol in BASE_UNIT_ALIASES:
        return ALL_UNITS[BASE_UNIT_ALIASES[symbol]]
    return None


def _parse_prefixed_unit(symbol: str) -> Tuple[Optional[UnitDefinition], float]:
    """
    Try to parse a prefixed unit like "km", "mN", "uA".

    Returns:
        (UnitDefinition, prefix_factor) or (None, 1.0) if not found.
    """
    # Try two-character prefixes first (like "da")
    for prefix_len in [2, 1]:
        if len(symbol) > prefix_len:
            prefix = symbol[:prefix_len]
            base = symbol[prefix_len:]

            if prefix in PREFIX_SYMBOLS and base in ALL_UNITS:
                power = PREFIX_SYMBOLS[prefix]
                return (ALL_UNITS[base], 10.0 ** power)

    return (None, 1.0)


# =============================================================================
# Unit Encoder
# =============================================================================

class UnitEncoder:
    """
    Encoder for physical units.

    Embedding structure (within 64-dim domain-specific region):
        dims 0-6:   SI exponents [M, L, T, I, Theta, N, J]
        dim 7:      Dimensionless flag
        dims 8-9:   Conversion factor to SI (sign, log|factor|)
        dim 10:     Is SI base unit flag
        dim 11:     Is derived SI unit flag
        dims 12-19: Unit identifier hash
        dims 20-27: Prefix encoding

    All conversions are deterministic given the encoded fields.
    """

    domain_tag = DOMAIN_TAGS["phys_unit"]
    domain_name = "phys_unit"

    def encode(self, unit_str: str) -> Any:
        """
        Encode a unit string.

        Args:
            unit_str: Unit string like "m", "m/s", "kg*m/s^2", "kN"

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        parsed = parse_unit(unit_str)

        emb = create_embedding()

        # Set domain tag (dims 0-7)
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Set dimension exponents (dims 8-14)
        dim_vec = parsed.dimensions.to_vector()
        for i in range(7):
            emb = backend.at_add(emb, 8 + i, dim_vec[i])

        # Dimensionless flag
        emb = backend.at_add(emb, 8 + UNIT_DIMENSIONLESS_FLAG, 
            1.0 if parsed.dimensions.is_dimensionless() else 0.0
        )

        # Conversion factor encoding
        if abs(parsed.total_factor - 1.0) < EPSILON:
            # Factor is 1.0 (SI base)
            emb = backend.at_add(emb, 8 + UNIT_CONV_SIGN, 1.0)
            emb = backend.at_add(emb, 8 + UNIT_CONV_LOG, 0.0)
        else:
            sign, log_mag = log_encode_value(parsed.total_factor)
            emb = backend.at_add(emb, 8 + UNIT_CONV_SIGN, sign)
            emb = backend.at_add(emb, 8 + UNIT_CONV_LOG, log_mag)

        # SI base/derived flags
        if len(parsed.base_units) == 1:
            symbol, exp = parsed.base_units[0]
            if exp == 1:
                if symbol in SI_BASE_UNITS:
                    emb = backend.at_add(emb, 8 + UNIT_IS_SI_BASE, 1.0)
                elif symbol in DERIVED_UNITS:
                    emb = backend.at_add(emb, 8 + UNIT_IS_SI_DERIVED, 1.0)

        # Unit identifier hash
        unit_hash = _simple_hash(unit_str, UNIT_ID_HASH_SIZE)
        for i, h in enumerate(unit_hash):
            emb = backend.at_add(emb, 8 + UNIT_ID_HASH_OFFSET + i, h)

        return emb

    def decode(self, emb: Any) -> str:
        """
        Decode an embedding back to a unit string.

        Note: This returns the canonical SI unit representation based on dimensions.
        The original unit string may not be exactly recovered if it used
        non-SI units or prefixes.

        Args:
            emb: 128-dim embedding

        Returns:
            Unit string (canonical SI form)
        """
        # Extract dimensions
        dims = self.decode_dimensions(emb)

        # Build canonical unit string from dimensions
        return self._dimensions_to_unit_string(dims)

    def decode_dimensions(self, emb: Any) -> Dimensions:
        """Extract dimensions from unit embedding."""
        dim_vec = emb[8:8 + 7]
        return Dimensions.from_vector(dim_vec)

    def decode_conversion_factor(self, emb: Any) -> float:
        """Extract the SI conversion factor from unit embedding."""
        sign = emb[8 + UNIT_CONV_SIGN].item()
        log_mag = emb[8 + UNIT_CONV_LOG].item()
        return log_decode_value(sign, log_mag)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid unit."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    def is_compatible(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if two units are compatible (same dimensions).

        Compatible units can be converted to each other.

        Args:
            emb1: First unit embedding
            emb2: Second unit embedding

        Returns:
            True if units have the same dimensions
        """
        backend = get_backend()
        dims1 = emb1[8:8 + 7]
        dims2 = emb2[8:8 + 7]
        return backend.allclose(dims1, dims2, atol=0.1).item()

    def get_conversion_factor(self, from_unit: str, to_unit: str) -> Optional[float]:
        """
        Get the conversion factor between two units.

        Args:
            from_unit: Source unit string
            to_unit: Target unit string

        Returns:
            Factor such that value_in_from_unit * factor = value_in_to_unit
            Returns None if units are incompatible.
        """
        from_emb = self.encode(from_unit)
        to_emb = self.encode(to_unit)

        if not self.is_compatible(from_emb, to_emb):
            return None

        # Factor = from_si_factor / to_si_factor
        from_factor = self.decode_conversion_factor(from_emb)
        to_factor = self.decode_conversion_factor(to_emb)

        return from_factor / to_factor

    def convert(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """
        Convert a value from one unit to another.

        Args:
            value: Numeric value
            from_unit: Source unit string
            to_unit: Target unit string

        Returns:
            Converted value, or None if units are incompatible.
        """
        factor = self.get_conversion_factor(from_unit, to_unit)
        if factor is None:
            return None
        return value * factor

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two units (e.g., m * s = m*s).

        Dimensions: Add exponents
        Conversion factor: Multiply

        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Add dimension exponents
        for i in range(7):
            result = backend.at_add(result, 8 + i, emb1[8 + i] + emb2[8 + i])

        # Update dimensionless flag
        is_dimensionless = all(
            abs(result[8 + i].item()) < 0.5 for i in range(7)
        )
        result = backend.at_add(result, 8 + UNIT_DIMENSIONLESS_FLAG, 
            1.0 if is_dimensionless else 0.0
        )

        # Multiply conversion factors (add logs)
        log1 = emb1[8 + UNIT_CONV_LOG].item()
        log2 = emb2[8 + UNIT_CONV_LOG].item()
        result = backend.at_add(result, 8 + UNIT_CONV_SIGN, 1.0)
        result = backend.at_add(result, 8 + UNIT_CONV_LOG, log1 + log2)

        return result

    def divide(self, emb1: Any, emb2: Any) -> Any:
        """
        Divide two units (e.g., m / s = m/s).

        Dimensions: Subtract exponents
        Conversion factor: Divide

        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Subtract dimension exponents
        for i in range(7):
            result = backend.at_add(result, 8 + i, emb1[8 + i] - emb2[8 + i])

        # Update dimensionless flag
        is_dimensionless = all(
            abs(result[8 + i].item()) < 0.5 for i in range(7)
        )
        result = backend.at_add(result, 8 + UNIT_DIMENSIONLESS_FLAG, 
            1.0 if is_dimensionless else 0.0
        )

        # Divide conversion factors (subtract logs)
        log1 = emb1[8 + UNIT_CONV_LOG].item()
        log2 = emb2[8 + UNIT_CONV_LOG].item()
        result = backend.at_add(result, 8 + UNIT_CONV_SIGN, 1.0)
        result = backend.at_add(result, 8 + UNIT_CONV_LOG, log1 - log2)

        return result

    def power(self, emb: Any, n: int) -> Any:
        """
        Raise a unit to an integer power (e.g., m^2).

        Dimensions: Multiply exponents by n
        Conversion factor: Raise to power

        """
        backend = get_backend()
        result = create_embedding()

        # Domain tag
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Multiply dimension exponents by n
        for i in range(7):
            result = backend.at_add(result, 8 + i, emb[8 + i] * n)

        # Update dimensionless flag
        is_dimensionless = all(
            abs(result[8 + i].item()) < 0.5 for i in range(7)
        )
        result = backend.at_add(result, 8 + UNIT_DIMENSIONLESS_FLAG, 
            1.0 if is_dimensionless else 0.0
        )

        # Power of conversion factor (multiply log by n)
        log_factor = emb[8 + UNIT_CONV_LOG].item()
        result = backend.at_add(result, 8 + UNIT_CONV_SIGN, 1.0)
        result = backend.at_add(result, 8 + UNIT_CONV_LOG, log_factor * n)

        return result

    def _dimensions_to_unit_string(self, dims: Dimensions) -> str:
        """Convert dimensions to canonical SI unit string."""
        numerator = []
        denominator = []

        # Map from dimension to base SI symbol
        dim_to_symbol = [
            ('M', 'kg'),
            ('L', 'm'),
            ('T', 's'),
            ('I', 'A'),
            ('Theta', 'K'),
            ('N', 'mol'),
            ('J', 'cd'),
        ]

        for attr, symbol in dim_to_symbol:
            exp = getattr(dims, attr)
            if exp > 0:
                if exp == 1:
                    numerator.append(symbol)
                else:
                    numerator.append(f"{symbol}^{exp}")
            elif exp < 0:
                if exp == -1:
                    denominator.append(symbol)
                else:
                    denominator.append(f"{symbol}^{-exp}")

        if not numerator and not denominator:
            return "1"  # Dimensionless

        num_str = "*".join(numerator) if numerator else "1"
        if denominator:
            den_str = "*".join(denominator)
            return f"{num_str}/{den_str}"
        return num_str

    def get_unit_info(self, unit_str: str) -> Dict:
        """
        Get detailed information about a unit.

        Args:
            unit_str: Unit string

        Returns:
            Dictionary with unit information
        """
        parsed = parse_unit(unit_str)
        emb = self.encode(unit_str)

        return {
            "original": unit_str,
            "dimensions": parsed.dimensions,
            "si_factor": parsed.total_factor,
            "components": parsed.base_units,
            "canonical_si": self._dimensions_to_unit_string(parsed.dimensions),
            "is_si_base": emb[8 + UNIT_IS_SI_BASE].item() > 0.5,
            "is_si_derived": emb[8 + UNIT_IS_SI_DERIVED].item() > 0.5,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

# Singleton encoder instance
_encoder: Optional[UnitEncoder] = None


def get_encoder() -> UnitEncoder:
    """Get the singleton UnitEncoder instance."""
    global _encoder
    if _encoder is None:
        _encoder = UnitEncoder()
    return _encoder


def encode_unit(unit_str: str) -> Any:
    """Convenience function to encode a unit."""
    return get_encoder().encode(unit_str)


def convert_units(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """Convenience function to convert between units."""
    return get_encoder().convert(value, from_unit, to_unit)


def are_compatible(unit1: str, unit2: str) -> bool:
    """Check if two units are compatible (can be converted)."""
    enc = get_encoder()
    emb1 = enc.encode(unit1)
    emb2 = enc.encode(unit2)
    return enc.is_compatible(emb1, emb2)


# =============================================================================
# Inline Tests
# =============================================================================

def _run_tests():
    """Run basic tests to verify the implementation."""
    encoder = UnitEncoder()

    print("Testing UnitEncoder...")

    # Test 1: Encode/decode base units
    print("\n1. Testing base units...")
    for unit in ["m", "kg", "s", "A", "K", "mol", "cd"]:
        emb = encoder.encode(unit)
        assert encoder.is_valid(emb), f"Invalid embedding for {unit}"
        dims = encoder.decode_dimensions(emb)
        print(f"   {unit}: {dims}")

    # Test 2: Encode/decode derived units
    print("\n2. Testing derived units...")
    for unit in ["N", "J", "W", "Pa", "Hz", "V", "C"]:
        emb = encoder.encode(unit)
        assert encoder.is_valid(emb), f"Invalid embedding for {unit}"
        dims = encoder.decode_dimensions(emb)
        canonical = encoder.decode(emb)
        print(f"   {unit}: {dims} -> {canonical}")

    # Test 3: Test unit parsing with expressions
    print("\n3. Testing unit expressions...")
    test_cases = [
        ("m/s", Dimensions(L=1, T=-1)),
        ("kg*m/s^2", Dimensions(M=1, L=1, T=-2)),
        ("J/s", Dimensions(M=1, L=2, T=-3)),
    ]
    for unit_str, expected_dims in test_cases:
        emb = encoder.encode(unit_str)
        dims = encoder.decode_dimensions(emb)
        assert dims == expected_dims, f"Wrong dims for {unit_str}: {dims} != {expected_dims}"
        print(f"   {unit_str}: {dims} OK")

    # Test 4: Test prefixed units
    print("\n4. Testing prefixed units...")
    prefixed = [("km", 1e3), ("mm", 1e-3), ("kN", 1e3), ("MHz", 1e6)]
    for unit, expected_factor in prefixed:
        emb = encoder.encode(unit)
        factor = encoder.decode_conversion_factor(emb)
        rel_error = abs(factor - expected_factor) / expected_factor
        assert rel_error < 0.01, f"Wrong factor for {unit}: {factor} != {expected_factor}"
        print(f"   {unit}: factor = {factor:.2e} OK")

    # Test 5: Test compatibility
    print("\n5. Testing compatibility...")
    compatible_pairs = [
        ("m", "km", True),
        ("m", "ft", True),
        ("m", "s", False),
        ("N", "kN", True),
        ("N", "J", False),
        ("J", "eV", True),
    ]
    for u1, u2, expected in compatible_pairs:
        emb1 = encoder.encode(u1)
        emb2 = encoder.encode(u2)
        result = encoder.is_compatible(emb1, emb2)
        assert result == expected, f"Wrong compatibility for {u1}, {u2}"
        print(f"   {u1} ~ {u2}: {result} OK")

    # Test 6: Test conversion
    print("\n6. Testing conversions...")
    conversions = [
        (1.0, "km", "m", 1000.0),
        (1.0, "h", "s", 3600.0),
        (1.0, "kN", "N", 1000.0),
        (100.0, "cm", "m", 1.0),
        (1.0, "eV", "J", 1.602176634e-19),
    ]
    for val, from_u, to_u, expected in conversions:
        result = encoder.convert(val, from_u, to_u)
        assert result is not None, f"Conversion failed for {from_u} -> {to_u}"
        rel_error = abs(result - expected) / expected if expected != 0 else abs(result)
        assert rel_error < 0.01, f"Wrong conversion: {val} {from_u} -> {result} {to_u} (expected {expected})"
        print(f"   {val} {from_u} = {result:.6g} {to_u} OK")

    # Test 7: Test unit multiplication
    print("\n7. Testing unit operations...")
    m = encoder.encode("m")
    s = encoder.encode("s")
    m_per_s = encoder.divide(m, s)
    dims = encoder.decode_dimensions(m_per_s)
    assert dims == Dimensions(L=1, T=-1), f"m/s wrong: {dims}"
    print(f"   m / s = {dims} OK")

    m_squared = encoder.power(m, 2)
    dims = encoder.decode_dimensions(m_squared)
    assert dims == Dimensions(L=2), f"m^2 wrong: {dims}"
    print(f"   m^2 = {dims} OK")

    print("\nAll tests passed!")


if __name__ == "__main__":
    _run_tests()
