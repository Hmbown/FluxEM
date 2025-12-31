# FluxEM

**Separate rule discovery from rule execution.**

When rules are known, execute exactly. When unknown, induce from examples.

## What's Here

| Module | What it does | Key result |
|--------|--------------|------------|
| `fluxem.arithmetic` | Algebraic embeddings for +, -, ×, ÷ | 100% OOD accuracy |
| `fluxem.compositional` | SCAN oracle baseline | 100% all splits |
| `scan_inducer` | Learn operators from examples | Works with 1-3 examples |

## The Insight

Neural networks fail at arithmetic and compositional generalization because they try to **discover** structure through gradient descent on tokens.

FluxEM separates concerns:
- **Rule execution**: When you know the rules, encode them. Exact results, guaranteed.
- **Rule discovery**: When you don't know the rules, induce them from examples. Then execute exactly.

## Installation

```bash
git clone https://github.com/Hmbown/FluxEM.git
cd FluxEM
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### Arithmetic (Exact)

```python
from fluxem import create_unified_model

model = create_unified_model()
model.compute("1847*392")   # -> 724024.0
model.compute("12345+67890") # -> 80235.0
model.compute("56088/123")   # -> 456.0
```

No training. Algebraic identities guarantee correctness.

### SCAN Oracle (Rule Execution)

```python
from fluxem import AlgebraicSCANSolver

solver = AlgebraicSCANSolver()
solver.solve("jump around right twice")
# -> 'I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP ...'
```

This is an **oracle baseline**—it executes known rules. The point: once rules are known, composition is trivial.

### Rule Inducer (Rule Discovery)

```python
from scan_inducer import induce_unary_operator

examples = [
    ("walk around right", ["I_TURN_RIGHT", "I_WALK"] * 4),
    ("run around right", ["I_TURN_RIGHT", "I_RUN"] * 4),
]

program = induce_unary_operator(examples, "around right")
# -> Repeat(4, Concat(RTURN, ACTION))

program.execute(["I_JUMP"])
# -> ['I_TURN_RIGHT', 'I_JUMP', 'I_TURN_RIGHT', 'I_JUMP', ...]
```

Learns the operator from 2 examples. Generalizes to unseen primitives perfectly.

## Results

### Arithmetic

| Operation | OOD Accuracy | Method |
|-----------|--------------|--------|
| Addition | 100% | Linear embedding homomorphism |
| Subtraction | 100% | Linear embedding homomorphism |
| Multiplication | 100% | Log-space homomorphism |
| Division | 100% | Log-space homomorphism |
| Powers/Roots | 100% | Scalar multiplication in log-space |

Tested ranges: Addition [-100000, 100000], Multiplication [10, 1000] × [10, 1000]

### SCAN Oracle Baseline

| Split | Oracle | Seq2Seq | What it shows |
|-------|--------|---------|---------------|
| addprim_jump | 100% | ~1% | Rule discovery failed, not composition |
| addprim_turn_left | 100% | ~1% | Same |
| length | 100% | ~14% | Length extrapolation trivial with known rules |
| simple | 100% | ~99% | Memorization works when examples cover space |

### Rule Induction

| Operator | Examples | Generalization |
|----------|----------|----------------|
| around right | 1-3 | 100% on held-out primitives |

## How It Works

### Arithmetic

Addition is vector addition:
```
embed(a) + embed(b) = embed(a + b)
```

Multiplication is addition in log-space:
```
log_embed(a) + log_embed(b) = log_embed(a × b)
```

These are algebraic identities, not learned patterns. See `docs/FORMAL_DEFINITION.md`.

### SCAN Oracle

Composition rules encoded directly. This is deterministic execution, not learning. See `docs/SCAN_BASELINE.md`.

### Rule Inducer

Program synthesis over a typed DSL:
1. Enumerate candidate programs (bottom-up, by size)
2. Prune by observational equivalence (signature hashing)
3. Filter to programs consistent with all examples
4. Return smallest (MDL principle)

The induced program is explicit and human-readable.

## Why This Matters

SCAN has been used for 7 years to show neural networks fail at compositional generalization. They get ~0-20% on the hard splits.

The diagnosis was incomplete. Neural networks don't fail at **composition**—they fail at **rule discovery** from limited examples.

Once you separate the problems:
- Rule discovery → program synthesis (works with 1-3 examples)
- Rule execution → algebraic encoding (works perfectly)

## Prior Work

| Approach | Method | Difference |
|----------|--------|------------|
| NALU (Trask, 2018) | Learned log/exp gates | No learned parameters here |
| xVal (Golkar, 2023) | Learned scaling direction | Fixed algebraic structure |
| Abacus (McLeish, 2024) | Positional digit encoding | Continuous, not character-level |
| Lake & Baroni (2018) | Identified SCAN problem | We separate discovery/execution |

## Limitations

- **Arithmetic**: Precision bounded by IEEE-754 float32 (see `docs/ERROR_MODEL.md`)
- **SCAN Oracle**: Requires known grammar (that's the point—it's a baseline)
- **Inducer**: Currently handles unary operators; binary operators not yet implemented
- **Not a general-purpose reasoning system**

## Documentation

- `docs/FORMAL_DEFINITION.md` - Mathematical specification
- `docs/ERROR_MODEL.md` - Precision guarantees and failure modes
- `docs/SCAN_BASELINE.md` - Why the oracle is a baseline, not a result

## The Name

**Flux**: From transformation-first thinking in music theory (Lewin's Generalized Interval Systems)
**EM**: Embedding Model

## Citation

```bibtex
@software{fluxem2025,
  title={FluxEM: Algebraic Embeddings with Separated Rule Discovery},
  author={Hunter Bown},
  year={2025},
  url={https://github.com/Hmbown/FluxEM}
}
```

## License

MIT
