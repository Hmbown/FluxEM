# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![CI](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Deterministic embeddings where algebraic operations become vector operations.**

```
encode(a) + encode(b) = encode(a + b)
```

No training. No weights. Structure by construction.

<p align="center">
  <img src="docs/demo.gif" alt="FluxEM demo" width="600">
</p>

→ [Project page](https://hmbown.github.io/FluxEM) · [The Vision](https://hmbown.github.io/FluxEM/vision.html)

---

## Quick Start

```bash
pip install fluxem
```

```python
from fluxem import create_unified_model

model = create_unified_model()
model.compute("12345 + 67890")  # → 80235.0
model.compute("144 * 89")       # → 12816.0
model.compute("10 m/s * 5 s")   # → 50.0 m
```

## What This Is

FluxEM encodes typed domain values into fixed-dimensional vectors where selected operations map to linear algebra:

- **Addition/subtraction**: `encode(a) + encode(b) = encode(a + b)`
- **Multiplication/division**: `log_encode(a) + log_encode(b) = log_encode(a × b)`

This extends across eleven domains—each with encoders that preserve algebraic structure.

| Domain | Example | Operations |
|--------|---------|------------|
| Physics | `9.8 m/s²` | Unit conversion, dimensional analysis |
| Chemistry | `C6H12O6` | Stoichiometry, mass balance |
| Biology | `ATGCCGTAG` | GC content, melting temp, translation |
| Math | `3 + 4i` | Complex numbers, matrices, vectors, polynomials |
| Logic | `p ∧ q → r` | Tautology detection, satisfiability |
| Music | `{0, 4, 7}` | Transposition, inversion, Forte numbers |
| Geometry | △ABC | Area, centroid, circumcenter |
| Graphs | G = (V, E) | Connectivity, cycles, shortest path |
| Sets | A ∪ B | Union, intersection, composition |
| Number Theory | 360 = 2³·3²·5 | Prime factorization, modular arithmetic |
| Data | [x₁, x₂, ...] | Arrays, records, tables |

## What This Is Not

- **Not symbolic math.** Won't simplify `x + x → 2x`. Use SymPy for that.
- **Not semantic embeddings.** Won't find "king − man + woman ≈ queen". Use text embeddings.
- **Not learned.** No parameters, no training, no drift.

## Domain Examples

### Polynomials as Vectors

Adding two embedding vectors is equivalent to adding the polynomials they represent:

```python
from fluxem.domains.math import PolynomialEncoder

enc = PolynomialEncoder(degree=2)

# P1: 3x² + 2x + 1
# P2:       x - 1
vec1 = enc.encode([3, 2, 1])
vec2 = enc.encode([0, 1, -1])

# The vector sum IS the polynomial sum
sum_vec = vec1 + vec2
print(enc.decode(sum_vec))
# → [3, 3, 0]  (which is 3x² + 3x)
```

### Music: Harmony as Geometry

In pitch-class set theory, transposition is vector addition:

```python
from fluxem.domains.music import ChordEncoder

enc = ChordEncoder()

# Encode a C Major chord
c_major = enc.encode("C", quality="major")

# Transpose by adding to the vector
f_major = enc.transpose(c_major, 5)  # Up 5 semitones

root, quality, _ = enc.decode(f_major)
print(f"{root} {quality}")
# → "F major"
```

### Logic: Tautologies by Construction

Propositional logic encoded such that valid formulas occupy specific subspaces:

```python
from fluxem.domains.logic import PropositionalEncoder, PropFormula

p = PropFormula.atom('p')
q = PropFormula.atom('q')

enc = PropositionalEncoder()
formula_vec = enc.encode(p.implies(q) | ~p)

print(enc.is_tautology(formula_vec))
# → True
```

<details>
<summary><strong>Where this idea comes from</strong></summary>

In the early twentieth century, Schoenberg and the Second Viennese School developed atonal theory—a framework that treated all twelve chromatic pitches as mathematically equal. This led to pitch-class set theory: reduce notes to numbers 0–11, analyze chords as sets.

Something unexpected emerged. The math revealed hidden structure in tonality itself.

Consider major and minor triads:

```
C major: {0, 4, 7}  intervals: 4, then 3
C minor: {0, 3, 7}  intervals: 3, then 4
```

These are **inversions** of each other—the same intervals, reversed. Their interval-class vector is identical: `(0, 0, 1, 1, 1, 0)`. In pitch-class set theory, they're the same object under a group action.

Paul Hindemith (1895–1963) took this further. He didn't reject atonality—he used its mathematical tools to rebuild tonal theory from first principles. The dichotomy dissolved: tonality and atonality were the same structure, viewed differently.

When you encode pitch-class sets into vector space with the algebra intact:
- **Transposition** becomes vector addition
- **Inversion** becomes a linear transformation
- The relationship between major and minor is geometric

The embedding doesn't represent the structure. It *is* the structure.

This extends across domains. The cyclic group of pitch transposition (Z₁₂) is isomorphic to clock arithmetic. The lattice of propositional logic mirrors set operations. If domains are embedded with their algebra intact, a model could recognize when structures are the same—not similar, but algebraically identical.

Read the full story: [Where this comes from](https://hmbown.github.io/FluxEM/vision.html)

</details>

## Reproducibility

```bash
git clone https://github.com/Hmbown/FluxEM.git && cd FluxEM
pip install -e ".[jax]"
python experiments/scripts/compare_embeddings.py
```

Outputs TSV tables comparing FluxEM to baselines:

```
table=accuracy_by_encoder
approach              dataset          exact_match    numeric_accuracy
FluxEM                id               1.000000       1.000000
FluxEM                ood_magnitude    1.000000       1.000000
Character             ood_magnitude    0.000000       0.012000
```

## Installation

```bash
pip install fluxem              # Core (NumPy)
pip install fluxem[jax]         # With JAX
pip install fluxem[mlx]         # With MLX (Apple Silicon)
pip install fluxem[full-jax]    # Full with HuggingFace
```

## Precision

| Operation | Relative Error (float32) |
|-----------|-------------------------|
| Add/Sub   | < 1e-7 |
| Mul/Div   | < 1e-6 |

Edge cases: `log(0)` → masked, division by zero → signed infinity, negative base with fractional exponent → unsupported.

See [ERROR_MODEL.md](docs/ERROR_MODEL.md) and [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md).

## Related Work

| Approach | Method | Difference |
|----------|--------|------------|
| [NALU](https://arxiv.org/abs/1808.00508) | Learned log/exp gates | FluxEM: no learned parameters |
| [xVal](https://arxiv.org/abs/2310.02989) | Learned scaling | FluxEM: deterministic, multi-domain |
| [Abacus](https://arxiv.org/abs/2405.17399) | Positional digits | FluxEM: algebraic structure |

## Citation

```bibtex
@software{fluxem2026,
  title={FluxEM: Algebraic Embeddings for Deterministic Neural Computation},
  author={Bown, Hunter},
  year={2026},
  url={https://github.com/Hmbown/FluxEM}
}
```

## License

MIT
