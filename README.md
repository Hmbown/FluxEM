# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Algebraic embeddings for arithmetic with IEEE-754 float precision.**

<p align="center">
  <img src="docs/demo.gif" alt="FluxEM terminal demo" width="600">
</p>

## Supported Operations

Arithmetic through algebraic embeddings — no training required.

| Operation | Syntax | Embedding |
|-----------|--------|-----------|
| Addition | `a + b` | Linear (vector addition) |
| Subtraction | `a - b` | Linear (vector subtraction) |
| Multiplication | `a * b` | Logarithmic (log-space addition) |
| Division | `a / b` | Logarithmic (log-space subtraction) |
| Powers | `a ** b` | Logarithmic (repeated multiplication) |
| Roots | `sqrt(a)` | Logarithmic (fractional power) |

All operations generalize out-of-distribution within IEEE-754 floating-point tolerance.
See [ERROR_MODEL.md](docs/ERROR_MODEL.md) for precision details.

## Installation

```bash
pip install fluxem
```

Or from source:

```bash
git clone https://github.com/Hmbown/FluxEM.git
cd FluxEM
pip install -e .
```

## Quick Start

```python
from fluxem import create_unified_model

model = create_unified_model()

model.compute("1234 + 5678")  # -> 6912.0
model.compute("250 * 4")      # -> 1000.0
model.compute("1000 / 8")     # -> 125.0
model.compute("3 ** 4")       # -> 81.0
```

### Extended API

```python
from fluxem import create_extended_ops
ops = create_extended_ops()
```

| Method | Example | Result |
|--------|---------|--------|
| `power(base, exp)` | `ops.power(2, 16)` | `65536.0` |
| `sqrt(x)` | `ops.sqrt(256)` | `16.0` |
| `exp(x)` | `ops.exp(1.0)` | `2.718...` |
| `ln(x)` | `ops.ln(2.718)` | `1.0...` |

## How It Works

| Embedding | Operations | Property | Example |
|-----------|------------|----------|---------|
| **Linear** | `+` `-` | Vector addition = arithmetic addition | `embed(3) + embed(5) = embed(8)` |
| **Logarithmic** | `*` `/` `**` | Log-space addition = multiplication | `log(3) + log(4) = log(12)` |

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) for mathematical details.

## The Insight

This approach comes from music theory. Lewin's Generalized Interval Systems (1987) formalized how musical intervals form a group structure. FluxEM applies the same framework:

| GIS Component | Music Theory | FluxEM |
|---------------|--------------|--------|
| **S** (space) | Pitches | ℝ (numbers) |
| **IVLS** (intervals) | ℤ₁₂ (semitones) | ℝ under + |
| **int(a,b)** | Pitch distance | Embedding distance |

## Prior Work

| Approach | Method | FluxEM Difference |
|----------|--------|-------------------|
| NALU (Trask, 2018) | Learned log/exp gates | No learned parameters |
| xVal (Golkar, 2023) | Learned scaling direction | Fixed algebraic structure |
| Abacus (McLeish, 2024) | Positional digit encoding | Continuous embeddings |

## Limitations

| Constraint | Reason |
|------------|--------|
| IEEE-754 precision | Floating-point bounds error |
| Zero handling | log(0) undefined |
| Sign tracked separately | Log-space is magnitude-only |
| Arithmetic only | Not a general reasoning system |

## Citation

```bibtex
@software{fluxem2025,
  title={FluxEM: Algebraic Embeddings for Neural Arithmetic},
  author={Bown, Hunter},
  year={2025},
  url={https://github.com/Hmbown/FluxEM}
}
```

## License

MIT
