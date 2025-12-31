# FluxEM Benchmark Results

## Summary

| Method | ID Test | OOD-A | OOD-B | OOD-C | OOD Avg |
|--------|---------|-------|-------|-------|---------|
| FluxEM (ours) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Transformer | 1.5% | 0.0% | 0.5% | 2.0% | 0.8% |
| GRU | 0.0% | 0.5% | 0.5% | 0.5% | 0.5% |

## Training Details

- **Training data**: 10K expressions, integers [0, 999], 1-3 operations
- **Epochs**: 50
- **Transformer**: ~72K params, d_model=64, 4 heads, 2 layers
- **GRU**: ~47K params, embed=32, hidden=64, bidirectional

## Notes

- **Accuracy**: Fraction of predictions within 1% relative error of ground truth
- **OOD-A**: Large integers [10K, 1M]
- **OOD-B**: Longer expressions (4-8 operations)
- **OOD-C**: Mixed operations with exponentiation

## Key Finding

FluxEM maintains 100% accuracy across all distributions because arithmetic
is guaranteed by structure-preserving embeddings (homomorphisms), not learned.

The baselines are intentionally small to isolate structure from scale. Larger
models with more data would likely improve, but cannot provide the same guarantee.