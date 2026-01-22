"""Domain modules for FluxEM Tools.

Each domain module provides a set of deterministic computation tools.
Tools are automatically registered when the module is imported.

Domains:
    - arithmetic: Basic arithmetic operations
    - physics: Units, dimensions, conversions
    - chemistry: Molecular calculations
    - biology: DNA, protein analysis
    - math_advanced: Complex numbers, vectors, matrices
    - music: Music theory, scales, intervals
    - geometry: Shapes, distances, transforms
    - graphs: Network analysis
    - sets: Set operations
    - logic: Propositional and predicate logic
    - number_theory: Primes, modular arithmetic
    - data: Arrays, records, tables
    - combinatorics: Permutations, combinations
    - probability: Distributions, Bayes
    - statistics: Descriptive statistics
    - information: Entropy, information theory
    - signal: Signal processing
    - calculus: Derivatives, integrals
    - temporal: Date/time calculations
    - finance: Interest, NPV, payments
    - optimization: Gradient descent, least squares
    - control: Control systems
    - color: Color space conversions
    - geospatial: Geographic calculations
    - geometric_algebra: Clifford algebra
    - security: Permission checking
    - electrical: Ohm's law, circuits (NEW)
    - thermodynamics: Heat, gas laws (NEW)
    - acoustics: Sound, decibels (NEW)
    - astronomy: Orbital mechanics (NEW)
    - pharmacology: Drug kinetics (NEW)
    - fluid_dynamics: Flow, pressure (NEW)
    - optics: Lenses, light (NEW)
    - nuclear: Decay, radiation (NEW)
    - cooking: Recipe scaling (NEW)
    - currency: Exchange rates (NEW)
    - fitness: Health calculations (NEW)
    - travel: Time zones, fuel (NEW)
    - diy: Construction calcs (NEW)
    - photography: Exposure, DoF (NEW)
    - gardening: Soil, plants (NEW)
    - text: String distances, readability (NEW)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..registry import ToolRegistry

# List of all domain modules to load
DOMAIN_MODULES = [
    # Existing domains (from current tool_registry.py)
    "arithmetic",
    "physics",
    "chemistry",
    "biology",
    "math_advanced",
    "music",
    "geometry",
    "graphs",
    "sets",
    "logic",
    "number_theory",
    "data",
    "combinatorics",
    "probability",
    "statistics",
    "information_theory",
    "signal_processing",
    "calculus",
    "temporal",
    "finance",
    "optimization",
    "control_systems",
    "color",
    "geospatial",
    "geometric_algebra",
    "security",
    # New scientific domains
    "electrical",
    "thermodynamics",
    "acoustics",
    "astronomy",
    "pharmacology",
    "fluid_dynamics",
    "optics",
    "nuclear",
    # New practical domains
    "cooking",
    "currency",
    "fitness",
    "travel",
    "diy",
    "photography",
    "gardening",
    # Text/linguistics
    "text",
]


def register_all(registry: "ToolRegistry") -> None:
    """Register all tools from all domain modules.

    This function is called automatically when get_registry() is called.
    It imports each domain module and registers its tools.

    Args:
        registry: The ToolRegistry to register tools in
    """
    import importlib
    import sys

    for module_name in DOMAIN_MODULES:
        try:
            # Import the domain module
            full_name = f"fluxem_tools.domains.{module_name}"
            if full_name in sys.modules:
                module = sys.modules[full_name]
            else:
                module = importlib.import_module(f".{module_name}", package="fluxem_tools.domains")

            # Call register_tools if it exists
            if hasattr(module, "register_tools"):
                module.register_tools(registry)
        except ImportError as e:
            # Domain not yet implemented - skip silently
            # This allows incremental development
            pass
        except Exception as e:
            # Log other errors but don't fail
            import warnings
            warnings.warn(f"Error loading domain {module_name}: {e}")


def get_available_domains() -> list:
    """Get list of domains that are currently implemented.

    Returns:
        List of domain names that can be loaded
    """
    import importlib

    available = []
    for module_name in DOMAIN_MODULES:
        try:
            importlib.import_module(f".{module_name}", package="fluxem_tools.domains")
            available.append(module_name)
        except ImportError:
            pass
    return available
