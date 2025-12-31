# SCAN Oracle Baseline

## What This Is

An **oracle baseline** for the SCAN benchmark that encodes the compositional semantics directly. This is NOT a learning result--it's a diagnostic that separates rule discovery from rule execution.

## What It Shows

| Split | Oracle Accuracy | Seq2Seq | What This Means |
|-------|-----------------|---------|-----------------|
| addprim_jump | 100% | ~1% | Seq2seq failed to discover the rule for "jump" compositions |
| addprim_turn_left | 100% | ~1% | Same: rule discovery failure |
| length | 100% | ~14% | Length extrapolation is trivial once rules are known |
| simple | 100% | ~99% | When examples cover the space, seq2seq memorizes fine |

## The Diagnostic

SCAN benchmarks conflate two capabilities:
1. **Rule discovery**: Inferring composition rules from examples
2. **Rule execution**: Applying known rules to new inputs

Neural networks fail at (1), not (2). Once rules are known, composition is trivial--as this oracle demonstrates.

## What This Does NOT Show

- That neural networks are "bad at composition" (they're bad at rule induction from limited data)
- That hand-coding beats learning (we're not proposing to hand-code everything)
- That this approach scales to natural language (SCAN is synthetic and unambiguous)

## The Research Direction

The interesting question is: **Can we learn the rules from limited examples?**

Milestones:
1. Few-shot lexicon induction (given parse tree, infer operator meanings)
2. Grammar + lexicon induction (infer both)
3. Noise/ambiguity robustness

This oracle baseline is step 0: verify that rule execution is trivial.

## Implementation

```python
from fluxem.compositional import AlgebraicSCANSolver

solver = AlgebraicSCANSolver()
solver.solve("jump around right twice")
# -> "I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP ..." (deterministic)
```

~100 lines of Python. No learning. No parameters.

## The Encoded Rules

The solver encodes SCAN's compositional semantics:

**Primitives:**
- `walk` -> `I_WALK`
- `run` -> `I_RUN`
- `jump` -> `I_JUMP`
- `look` -> `I_LOOK`

**Modifiers:**
- `left` -> prepend `I_TURN_LEFT`
- `right` -> prepend `I_TURN_RIGHT`
- `opposite` -> prepend direction turn twice
- `around` -> repeat (turn + action) four times

**Repetition:**
- `twice` -> duplicate output
- `thrice` -> triplicate output

**Composition:**
- `and` -> concatenate outputs
- `after` -> reverse order then concatenate

These rules are the complete SCAN grammar. Once encoded, 100% accuracy is guaranteed.

## Why This Matters

This baseline demonstrates the **separation thesis**:

> When the rule system is known (or cheaply recoverable), bake it into the representation and stop wasting model capacity relearning algebra/grammar.

The SCAN oracle and FluxEM arithmetic embeddings illustrate the same principle:
- Arithmetic: known algebraic structure -> encode as geometry
- SCAN: known compositional rules -> encode as algebraic transformations

Neither requires learning. Both achieve 100% generalization.

## References

- Lake & Baroni (2018). "Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks"
- Original SCAN dataset: https://github.com/brendenlake/SCAN
