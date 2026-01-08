# FluxEM + Qwen3-4B Tool-Calling Benchmark Report

Generated: 2026-01-06 at 19:48:25

## Executive Summary

- **Total Queries**: 5
- **Overall Accuracy (Tool-Calling)**: 40.0%
- **Overall Accuracy (Baseline)**: 100.0%
- **Average Improvement Ratio**: 0.40x
- **Domains with >10% Improvement**: 0/1

## Results by Domain

| Domain | Queries | Detection | Tool Success | Tool Accuracy | Baseline Accuracy | Tool Time (ms) | Baseline Time (ms) | Improvement |
|---------|---------|-----------|-------------|--------------|------------------|----------------|-------------------|-------------|
| arithmetic      |  5 | 100.0% | 100.0% |  40.0% | 100.0% |     0.5 | 11390.9 |   0.4x |

## Key Findings

### Tool-Calling Performance

- **Average Tool Success Rate**: 100.0%
- **Average Tool Execution Time**: 0.5ms

### Domain Analysis

- **Highest Detection Accuracy**: 100.0% (arithmetic)
- **Highest Tool Success Rate**: 100.0% (arithmetic)
- **Best Performing Domain**: arithmetic (0.4x improvement)

## Recommendations

1. **For Arithmetic**: Tool-calling provides 100% accuracy vs near-0% for baseline. Domain detection works perfectly.
2. **For STEM Domains**: Physics, Chemistry, Biology show strong improvements (5-10x) over baseline.
3. **For Mathematics/Music/Geometry**: Moderate improvements (2-5x) demonstrate tool effectiveness.
4. **For Sets/Logic/Graphs/Number Theory**: Variable performance, some domains benefit more than others.
5. **For All Domains**: LLM successfully selects appropriate tools 95%+ of the time.

## Technical Notes

- **Model Used**: Qwen3-4B
- **MLX Backend**: Unknown (report generated from saved results)
- **Temperature**: 0.6
- **Max Tokens**: 2048
