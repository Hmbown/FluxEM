# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![CI](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Turn structure into vectors. Algebraically.**

> "What if `vector(A) + vector(B)` wasn't just a similarity search, but the actual result of `A + B`?"

FluxEM provides **deterministic embeddings** where algebraic operations on data become linear operations in vector space. No training, no weights, no hallucinations—just structure by construction.

<p align="center">
  <img src="docs/demo.gif" alt="FluxEM demo" width="600">
</p>

→ [Project page](https://hmbown.github.io/FluxEM) · [The Vision](https://hmbown.github.io/FluxEM/vision.html)

---

## ⚡️ Quick Start

You can use FluxEM as a "neuro-symbolic calculator" right out of the box.

```bash
pip install fluxem
```

```python
from fluxem import create_unified_model

# It looks like a calculator, but under the hood, this is vector algebra.
model = create_unified_model()

# Arithmetic is just vector addition
model.compute("123 + 456")      # → 579.0

# Even complex dimensional analysis works
model.compute("10 m/s * 5 s")   # → 50.0 m
```

## 🧠 The Big Idea

Most embeddings (like word2vec or BERT) are **semantic**: they learn that *King* is close to *Queen*.
FluxEM embeddings are **algebraic**: they encode the *rules* of the domain itself.

```
encode(x) + encode(y) = encode(x + y)
```

This means you can perform symbolic reasoning using nothing but neural network primitives (vector addition, matrix multiplication).

### Why is this useful?
1. **Grounding for LLMs**: Give your models a "native" understanding of math, logic, and science without tokenization hacks.
2. **Infinite Data**: Generate millions of algebraically consistent training examples for your own models.
3. **Zero-Shot Structure**: No need to train an encoder to understand that `3+4=7`. It's built in.

## 📐 Examples: See the Math in Action

The "Algebra" isn't just for numbers. It applies to structure.

### 1. Polynomials as Vectors
In the math domain, adding two embedding vectors is equivalent to adding the polynomials they represent.

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

### 2. Music Theory: Harmony is Geometry
This is where it gets wild. In pitch-class set theory, "transposition" (moving a chord up via pitch) is just vector addition.

```python
from fluxem.domains.music import ChordEncoder

enc = ChordEncoder()

# Encode a C Major chord
c_major = enc.encode("C", quality="major")

# Transpose it by adding to the vector
# (This isn't a helper function, it's literal vector math)
f_major = enc.transpose(c_major, 5)  # Up 5 semitones

root, quality, _ = enc.decode(f_major)
print(f"{root} {quality}")
# → "F major"
```

### 3. Logic: Tautologies by Definition
We can encode propositional logic such that "True" formulas land in a specific subspace.

```python
from fluxem.domains.logic import PropositionalEncoder, PropFormula

p = PropFormula.atom('p')
q = PropFormula.atom('q')

# Formula: (p implies q) OR (not p) used to check tautology
# This is physically encoded into the vector structure
enc = PropositionalEncoder()
formula_vec = enc.encode(p.implies(q) | ~p)

print(enc.is_tautology(formula_vec))
# → True
```

## 🌍 Supported Domains

We support 11 domains where structure is preserved:

| Domain | What gets preserved? |
|--------|----------------------|
| **Physics** | Units, dimensions, physical constants |
| **Chemistry** | Stoichiometry, molecule composition |
| **Biology** | DNA/RNA sequences, codons, melting temp |
| **Math** | Complex numbers, polynomials, rational numbers |
| **Music** | Pitch classes, intervals, chord inversions |
| **Logic** | Propositional logic, simple predicates |
| **Geometry** | Shapes, areas, centroids |
| **Sets** | Union, intersection, subsets |
| **Graphs** | Adjacency, simple cycles |
| **Number Theory** | Prime factorization, modular arithmetic |
| **Data** | Structured records, schema validation |

## 🛠 Installation

```bash
pip install fluxem              # Core (NumPy only)
pip install "fluxem[jax]"       # With JAX (recommended for speed)
pip install "fluxem[mlx]"       # With MLX (for Apple Silicon)
```

## ⚠️ Limitations

*   **It's NOT a symbolic solver**: It won't simplify `sin(x)^2 + cos(x)^2` to `1` unless specifically encoded. It preserves structure, it doesn't perform arbitrary reduction.
*   **Precision matters**: We implement a custom error model to track floating point drift. For complex chains of operations, error can accumulate (~1e-7).
*   **Fixed Dimensions**: Embeddings are fixed size (default 128 or 256), so there is a limit to the complexity (e.g., number of polynomial terms) you can encode before "overflowing".

## Citation

If you use FluxEM in your research, please cite:

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
