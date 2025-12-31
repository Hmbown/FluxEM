# FluxEM

**Algebraic embeddings for arithmetic with IEEE-754 float precision.**

<p align="center">
  <img src="docs/demo.gif" alt="FluxEM terminal demo" width="600">
</p>

## The Problem

Neural networks often struggle with arithmetic. They can explain calculus but fumble `1847 × 392`. This happens because they treat numbers as arbitrary tokens.

## The Solution

FluxEM encodes numbers so that arithmetic operations become geometric operations:

- **Addition**: `embed(a) + embed(b) = embed(a + b)`
- **Multiplication**: `log_embed(a) + log_embed(b) = log_embed(a × b)`

Systematic generalization via algebraic structure (no learned parameters).

## Results

| Operation | OOD Accuracy | Training Required |
|-----------|--------------|-------------------|
| Addition | Within tolerance | None |
| Subtraction | Within tolerance | None |
| Multiplication | Within tolerance | None |
| Division | Within tolerance | None |
| Powers | Within tolerance | None |
| Roots | Within tolerance | None |

Tested on ranges outside any "training" distribution. See [ERROR_MODEL.md](docs/ERROR_MODEL.md) for precision details.

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

Extended operations:

```python
from fluxem import create_extended_ops

ops = create_extended_ops()

ops.power(2, 16)  # -> 65536.0
ops.sqrt(256)     # -> 16.0
ops.exp(1.0)      # -> 2.718...
ops.ln(2.718)     # -> 1.0...
```

## How It Works

### Linear Embeddings (Addition)

Numbers are embedded as vectors along a fixed direction. Vector addition in embedding space equals arithmetic addition.

```
embed(3) + embed(5) = embed(8)  # By design
```

### Log Embeddings (Multiplication)

Numbers are embedded in log-space. Addition in log-space equals multiplication in linear space.

```
log_embed(3) + log_embed(4) = log_embed(12)  # By design
```

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) for mathematical details.

## The Insight

This approach comes from music theory. Lewin's Generalized Interval Systems (1987) formalized how musical intervals form a group structure. FluxEM applies the same framework to numbers:

- S = ℝ (numbers, like musical pitches)
- IVLS = ℝ under + (intervals between numbers)
- int(a, b) = embedding distance

The framework that unified 20th-century music theory informs this arithmetic embedding design.

## Prior Work

| Approach | Method | Difference |
|----------|--------|------------|
| NALU (Trask, 2018) | Learned log/exp gates | No learned parameters here |
| xVal (Golkar, 2023) | Learned scaling direction | Fixed algebraic structure |
| Abacus (McLeish, 2024) | Positional digit encoding | Continuous, not character-level |

## Limitations

- Precision bounded by IEEE-754 floating point
- Zero requires special handling (log(0) undefined)
- Sign tracked separately from magnitude in log space
- Not a general reasoning system—just arithmetic

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
