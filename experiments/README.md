# FluxEM Experiments

Minimal end-to-end experiments comparing token-only vs hybrid (FluxEM) training.

## Quick Start

```bash
# Generate data, train both models, evaluate
python scripts/generate_data.py --config configs/arithmetic_small.yaml
python scripts/train_token_only.py --config configs/arithmetic_small.yaml
python scripts/train_hybrid.py --config configs/arithmetic_small.yaml
python scripts/eval.py --config configs/arithmetic_small.yaml
```

## Structure

```
experiments/
├── configs/           # YAML configurations
├── scripts/           # Python scripts
├── data/              # Generated datasets (gitignored)
└── results/           # Model outputs (gitignored)
```

## Experiments

| Config | Task | Train Size | Time (CPU) |
|--------|------|------------|------------|
| `arithmetic_small.yaml` | Arithmetic | 1K | ~2 min |
| `arithmetic_full.yaml` | Arithmetic | 10K | ~20 min |
| `units_full.yaml` | Dimensional analysis | 5K | ~15 min |

## Expected Results

Hybrid models show near-perfect OOD generalization on arithmetic/units tasks
where token-only models fail (< 25% accuracy).

See `docs/EXPERIMENTS.md` for detailed documentation.
