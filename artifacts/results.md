# FluxEM Benchmark Results

## Summary

| Method | ID Test | OOD-A | OOD-B | OOD-C | OOD Avg |
|--------|---------|-------|-------|-------|---------|
| FluxEM (ours) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Transformer | 2.0% | 1.0% | 0.0% | 0.5% | 0.5% |
| GRU | 3.0% | 0.5% | 0.5% | 0.0% | 0.3% |

## Notes

- **Accuracy**: Fraction of predictions within 1% relative error of ground truth
- **OOD-A**: Large integers [10K, 1M]
- **OOD-B**: Longer expressions (4-8 operations)
- **OOD-C**: Mixed operations with exponentiation

## Key Finding

FluxEM maintains ~100% accuracy across all distributions because arithmetic
is implemented via structure-preserving embeddings (homomorphisms), not learned.
Baselines fail on OOD because they memorize patterns, not arithmetic structure.