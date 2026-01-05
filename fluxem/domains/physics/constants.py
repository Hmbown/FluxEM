"""
Physical Constants with Dimensions.

Pre-computed embeddings for fundamental physical constants.
All values are deterministic lookup - no learning.
"""

from typing import Any, Dict, Optional
from ...backend import get_backend

from .dimensions import DimensionalQuantity, Dimensions


class PhysicalConstants:
    """
    Database of physical constants as pre-computed embeddings.

    All constants are stored with their SI values and dimensions.
    """

    def __init__(self):
        backend = get_backend()
        self.encoder = DimensionalQuantity()
        self._constants: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict] = {}
        self._build_constants()

    def _build_constants(self) -> None:
        """Build the constants database."""

        # Speed of light
        self._add_constant(
            name="c",
            value=299_792_458.0,
            dimensions=Dimensions(L=1, T=-1),
            description="Speed of light in vacuum",
            uncertainty=0.0  # Exact by definition
        )

        # Planck constant
        self._add_constant(
            name="h",
            value=6.62607015e-34,
            dimensions=Dimensions(M=1, L=2, T=-1),
            description="Planck constant",
            uncertainty=0.0  # Exact by definition
        )

        # Reduced Planck constant
        self._add_constant(
            name="hbar",
            value=1.054571817e-34,
            dimensions=Dimensions(M=1, L=2, T=-1),
            description="Reduced Planck constant (h/2π)",
            uncertainty=0.0
        )

        # Gravitational constant
        self._add_constant(
            name="G",
            value=6.67430e-11,
            dimensions=Dimensions(M=-1, L=3, T=-2),
            description="Gravitational constant",
            uncertainty=1.5e-15
        )

        # Elementary charge
        self._add_constant(
            name="e",
            value=1.602176634e-19,
            dimensions=Dimensions(I=1, T=1),
            description="Elementary charge",
            uncertainty=0.0  # Exact by definition
        )

        # Electron mass
        self._add_constant(
            name="m_e",
            value=9.1093837015e-31,
            dimensions=Dimensions(M=1),
            description="Electron mass",
            uncertainty=2.8e-40
        )

        # Proton mass
        self._add_constant(
            name="m_p",
            value=1.67262192369e-27,
            dimensions=Dimensions(M=1),
            description="Proton mass",
            uncertainty=5.1e-37
        )

        # Boltzmann constant
        self._add_constant(
            name="k_B",
            value=1.380649e-23,
            dimensions=Dimensions(M=1, L=2, T=-2, Theta=-1),
            description="Boltzmann constant",
            uncertainty=0.0  # Exact by definition
        )

        # Avogadro constant
        self._add_constant(
            name="N_A",
            value=6.02214076e23,
            dimensions=Dimensions(N=-1),
            description="Avogadro constant",
            uncertainty=0.0  # Exact by definition
        )

        # Vacuum permittivity
        self._add_constant(
            name="epsilon_0",
            value=8.8541878128e-12,
            dimensions=Dimensions(M=-1, L=-3, T=4, I=2),
            description="Vacuum electric permittivity",
            uncertainty=1.3e-21
        )

        # Vacuum permeability
        self._add_constant(
            name="mu_0",
            value=1.25663706212e-6,
            dimensions=Dimensions(M=1, L=1, T=-2, I=-2),
            description="Vacuum magnetic permeability",
            uncertainty=1.9e-16
        )

        # Fine structure constant (dimensionless)
        self._add_constant(
            name="alpha",
            value=7.2973525693e-3,
            dimensions=Dimensions(),  # Dimensionless
            description="Fine structure constant",
            uncertainty=1.1e-12
        )

        # Stefan-Boltzmann constant
        self._add_constant(
            name="sigma",
            value=5.670374419e-8,
            dimensions=Dimensions(M=1, T=-3, Theta=-4),
            description="Stefan-Boltzmann constant",
            uncertainty=0.0
        )

        # Gas constant
        self._add_constant(
            name="R",
            value=8.314462618,
            dimensions=Dimensions(M=1, L=2, T=-2, Theta=-1, N=-1),
            description="Molar gas constant",
            uncertainty=0.0
        )

        # Standard gravity
        self._add_constant(
            name="g",
            value=9.80665,
            dimensions=Dimensions(L=1, T=-2),
            description="Standard acceleration due to gravity",
            uncertainty=0.0  # By definition
        )

        # Atomic mass unit
        self._add_constant(
            name="u",
            value=1.66053906660e-27,
            dimensions=Dimensions(M=1),
            description="Unified atomic mass unit",
            uncertainty=5.0e-37
        )

    def _add_constant(
        self,
        name: str,
        value: float,
        dimensions: Dimensions,
        description: str,
        uncertainty: float = 0.0
    ) -> None:
        """Add a constant to the database."""
        self._constants[name] = self.encoder.encode(value, dimensions)
        self._metadata[name] = {
            "value": value,
            "dimensions": dimensions,
            "description": description,
            "uncertainty": uncertainty,
        }

    def get(self, name: str) -> Optional[Any]:
        """Get the embedding for a physical constant."""
        return self._constants.get(name)

    def get_value(self, name: str) -> Optional[float]:
        """Get the numeric value of a constant."""
        meta = self._metadata.get(name)
        return meta["value"] if meta else None

    def get_dimensions(self, name: str) -> Optional[Dimensions]:
        """Get the dimensions of a constant."""
        meta = self._metadata.get(name)
        return meta["dimensions"] if meta else None

    def get_description(self, name: str) -> Optional[str]:
        """Get the description of a constant."""
        meta = self._metadata.get(name)
        return meta["description"] if meta else None

    def list_constants(self) -> list[str]:
        """List all available constant names."""
        return list(self._constants.keys())

    def search(self, dimensions: Dimensions) -> list[str]:
        """Find constants with matching dimensions."""
        matches = []
        for name, meta in self._metadata.items():
            if meta["dimensions"] == dimensions:
                matches.append(name)
        return matches


# Singleton instance
CONSTANTS = PhysicalConstants()


def get_constant(name: str) -> Optional[Any]:
    """Convenience function to get a constant embedding."""
    return CONSTANTS.get(name)
