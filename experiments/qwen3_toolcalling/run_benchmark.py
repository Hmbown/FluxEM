"""
Main Benchmark Script for FluxEM + Qwen3-4B Tool-Calling.

Orchestres running benchmarks comparing tool-calling performance
against baseline LLM without tools across FluxEM domains.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(1, str(Path(__file__).parent.parent.parent))


from .qwen3_wrapper import create_wrapper, Qwen3MLXWrapper
from .benchmark_data import BENCHMARK_PROMPTS
from .evaluator import create_evaluator, Evaluator, ToolCallResult
from .baselines import TransformersBaseline, WrapperBaseline, BaselineRunner


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FluxEM + Qwen3-4B Tool-Calling Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/.mlx/models/Qwen/Qwen3-4B-Instruct-MLX",
        help="Path to Qwen3-4B MLX model",
    )
    
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline evaluation (LLM without tools)",
    )

    parser.add_argument(
        "--baseline-backend",
        choices=["auto", "mlx", "transformers", "none"],
        default="auto",
        help="Baseline backend to use (default: auto)",
    )

    parser.add_argument(
        "--baseline-model-path",
        type=str,
        default=None,
        help="Local model path for transformers baseline",
    )

    parser.add_argument(
        "--baseline-tokenizer-path",
        type=str,
        default=None,
        help="Optional tokenizer path for transformers baseline",
    )

    parser.add_argument(
        "--baseline-device",
        type=str,
        default="cpu",
        help="Device for transformers baseline (cpu/cuda/mps)",
    )

    parser.add_argument(
        "--baseline-max-tokens",
        type=int,
        default=256,
        help="Max new tokens for baseline generation",
    )

    parser.add_argument(
        "--baseline-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for baseline generation",
    )

    parser.add_argument(
        "--baseline-trust-remote-code",
        action="store_true",
        help="Allow trust_remote_code when loading transformers model",
    )
    
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        help="Specific domains to test (default: all benchmarked domains)",
    )

    parser.add_argument(
        "--dataset",
        choices=["standard", "hard", "very_hard"],
        default="standard",
        help="Benchmark dataset to use (default: standard)",
    )

    parser.add_argument(
        "--tool-selection",
        choices=["pattern", "llm", "hybrid"],
        default="pattern",
        help="Tool selection strategy (default: pattern)",
    )

    parser.add_argument(
        "--llm-query-extraction",
        action="store_true",
        help="Use LLM to refine tool queries when selecting tools via LLM",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/qwen3_toolcalling/results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during benchmark",
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test on 3 prompts per selected domain",
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file",
    )
    
    return parser.parse_args()


def print_banner():
    """Print benchmark banner."""
    print("=" * 80)
    print("FluxEM + Qwen3-4B Tool-Calling Benchmark")
    print("=" * 80)
    print("Comparing LLM + Deterministic Tools vs LLM Alone")
    print("Testing FluxEM domains via deterministic tools")
    print("=" * 80)
    print()


def _to_serializable(value: Any) -> Any:
    """Convert values to JSON-serializable structures."""
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _create_baseline_runner(
    args, wrapper: Qwen3MLXWrapper
) -> Optional[BaselineRunner]:
    if args.no_baseline or args.baseline_backend == "none":
        return None

    if args.baseline_backend in ("auto", "mlx"):
        if wrapper.is_loaded:
            return WrapperBaseline(wrapper)
        if args.baseline_backend == "mlx":
            print("Warning: MLX baseline requested but model failed to load.")

    if args.baseline_backend in ("auto", "transformers"):
        if not args.baseline_model_path:
            if args.baseline_backend == "transformers":
                print("Warning: --baseline-model-path is required for transformers baseline.")
            return None
        try:
            return TransformersBaseline(
                model_path=args.baseline_model_path,
                tokenizer_path=args.baseline_tokenizer_path,
                device=args.baseline_device,
                max_new_tokens=args.baseline_max_tokens,
                temperature=args.baseline_temperature,
                trust_remote_code=args.baseline_trust_remote_code,
            )
        except Exception as exc:
            print(f"Warning: Failed to initialize transformers baseline: {exc}")
            return None

    return None


def _load_benchmark_prompts(dataset: str) -> Dict[str, Any]:
    if dataset == "hard":
        from .benchmark_data_hard import HARD_BENCHMARK_PROMPTS

        return HARD_BENCHMARK_PROMPTS
    if dataset == "very_hard":
        from .benchmark_data_very_hard import VERY_HARD_BENCHMARK_PROMPTS

        return VERY_HARD_BENCHMARK_PROMPTS
    return BENCHMARK_PROMPTS


def run_single_benchmark(
    wrapper: Qwen3MLXWrapper,
    evaluator: Evaluator,
    domain: str,
    prompt: str,
    expected: Any,
    tool_prompt: str,
    baseline_runner: Optional[BaselineRunner],
) -> Dict[str, Any]:
    """
    Run a single benchmark query.
    
    Returns:
        Dictionary with results
    """
    print(f"\n[{domain.upper()}] Testing: {prompt[:60]}...")
    
    # Run tool-calling version
    tool_response = wrapper.generate_with_tools(prompt)
    tool_time_ms = tool_response.get("execution_time_ms", 0.0)
    tool_result = ToolCallResult(
        domain=tool_response.get("domain", domain),
        tool_name=tool_response.get("tool_name") or "unknown",
        success=tool_response.get("tool_success", False),
        result=tool_response.get("result"),
        error=tool_response.get("error"),
        execution_time_ms=tool_time_ms,
    )
    
    # Run baseline version (if enabled)
    if baseline_runner is not None:
        baseline_result = baseline_runner.generate(prompt)
        baseline_response = baseline_result.response
        baseline_time_ms = baseline_result.time_ms
    else:
        baseline_response = None
        baseline_time_ms = None
    
    # Evaluate
    evaluation_result = evaluator.evaluate_response(
        prompt=prompt,
        tool_result=tool_result,
        baseline_response=baseline_response,
        expected_domain=domain,
        expected_answer=expected,
        tool_time_ms=tool_time_ms,
        baseline_time_ms=baseline_time_ms,
    )

    evaluator.add_result(evaluation_result)
    
    # Print result
    if evaluator.verbose:
        print(f"  Tool result: {tool_response.get('result', 'N/A')}")
        print(f"  Tool time: {tool_time_ms:.2f}ms")
        if baseline_response is not None:
            print(f"  Baseline result: {baseline_response[:60]}")
            print(f"  Baseline time: {baseline_time_ms:.2f}ms")
        print(f"  Correct: {evaluation_result.answer_correct}")
    
    return {
        "domain": domain,
        "prompt": prompt,
        "expected": expected,
        "tool_call": {
            "response": tool_response.get("result", None),
            "tool_name": tool_response.get("tool_name"),
            "tool_success": tool_response.get("tool_success", False),
            "time_ms": tool_time_ms,
        },
        "baseline": {
            "response": baseline_response,
            "time_ms": baseline_time_ms,
        },
        "evaluation": {
            "correct": evaluation_result.answer_correct,
            "baseline_correct": evaluation_result.baseline_correct,
            "domain_correct": evaluation_result.domain_correct,
            "tool_success": evaluation_result.tool_success,
        },
    }


def run_domain_benchmark(
    wrapper: Qwen3MLXWrapper,
    evaluator: Evaluator,
    domain: str,
    benchmark_prompts: Dict[str, Any],
    num_samples: Optional[int] = None,
    baseline_runner: Optional[BaselineRunner] = None,
) -> Dict[str, Any]:
    """
    Run benchmark on all prompts for a specific domain.
    
    Returns:
        Dictionary with domain metrics
    """
    print(f"\n{'=' * 80}")
    print(f"Running {domain.upper()} Domain Benchmark")
    print(f"{'=' * 80}")
    
    prompts_data = benchmark_prompts.get(domain, [])
    if not prompts_data:
        print(f"Warning: No prompts found for domain '{domain}'")
        return {
            "domain": domain,
            "total": 0,
            "correct": 0,
            "queries": [],
        }
    
    results = []
    sample_prompts = prompts_data if num_samples is None else prompts_data[:num_samples]
    for i, prompt_data in enumerate(sample_prompts):
        prompt = prompt_data["prompt"]
        expected = prompt_data["expected"]
        description = prompt_data.get("description", "N/A")
        
        print(f"\n[{i+1}/{len(sample_prompts)}] {description}")
        
        result = run_single_benchmark(
            wrapper=wrapper,
            evaluator=evaluator,
            domain=domain,
            prompt=prompt,
            expected=expected,
            tool_prompt=prompt,
            baseline_runner=baseline_runner,
        )
        
        results.append(result)
    
    # Calculate domain summary
    correct_count = sum(1 for r in results if r["evaluation"]["correct"])
    total_count = len(results)
    
    if total_count > 0:
        print(f"\n{domain.upper()} Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")
    else:
        print(f"\n{domain.upper()} Results: 0/0 correct (0.0%)")
    
    return {
        "domain": domain,
        "total": total_count,
        "correct": correct_count,
        "accuracy": (correct_count / total_count) if total_count else 0.0,
        "results": results,
    }


def run_full_benchmark(
    wrapper: Qwen3MLXWrapper,
    evaluator: Evaluator,
    args,
    baseline_runner: Optional[BaselineRunner],
    benchmark_prompts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run full benchmark across all domains in the selected dataset.
    
    Returns:
        Dictionary with all benchmark results
    """
    print_banner()
    
    # Determine which domains to test
    available_domains = list(benchmark_prompts.keys())
    available_set = set(available_domains)
    if args.domains:
        invalid = [d for d in args.domains if d not in available_set]
        if invalid:
            print(
                "Warning: requested domains not in dataset: "
                f"{', '.join(invalid)}"
            )
        test_domains = [d for d in args.domains if d in available_set]
        if not test_domains:
            raise ValueError(
                "No valid domains selected. Available: "
                + ", ".join(sorted(available_set))
            )
    else:
        test_domains = available_domains
    
    print(f"\nTesting domains: {', '.join(test_domains)}")
    print(f"Model path: {wrapper.model_path}")
    print(f"Dataset: {args.dataset}")
    baseline_enabled = baseline_runner is not None
    baseline_backend = baseline_runner.backend if baseline_runner is not None else "none"
    print(f"Baseline enabled: {baseline_enabled} ({baseline_backend})")
    print()
    
    # Run benchmarks
    all_results = {}
    
    for domain in test_domains:
        print(f"\n{'=' * 80}")
        print(f"Benchmarking {domain.upper()} Domain")
        print(f"{'=' * 80}")
        
        domain_result = run_domain_benchmark(
            wrapper=wrapper,
            evaluator=evaluator,
            domain=domain,
            benchmark_prompts=benchmark_prompts,
            num_samples=None,
            baseline_runner=baseline_runner,
        )
        
        all_results[domain] = domain_result
    
    # Generate and display report
    print(f"\n{'=' * 80}")
    print("Benchmark Complete!")
    print(f"{'=' * 80}")
    print()
    
    # Generate summary
    total_queries = sum(r["total"] for r in all_results.values())
    total_correct = sum(r["correct"] for r in all_results.values())
    overall_accuracy = total_correct / total_queries * 100 if total_queries else 0.0
    metrics = evaluator.aggregate_metrics()
    baseline_totals = [
        (m.baseline_accuracy, m.total_queries)
        for m in metrics.values()
        if m.baseline_accuracy is not None
    ]
    if baseline_totals and total_queries:
        baseline_total = sum(acc * total / 100 for acc, total in baseline_totals)
        overall_baseline_accuracy = baseline_total / total_queries * 100
    else:
        overall_baseline_accuracy = None
    
    print(f"Overall Results:")
    print(f"  Total queries: {total_queries}")
    print(f"  Total correct: {total_correct}")
    print(f"  Overall accuracy: {overall_accuracy:.1f}%")
    if overall_baseline_accuracy is not None:
        print(f"  Overall baseline accuracy: {overall_baseline_accuracy:.1f}%")
    print()
    
    # Save results if requested
    if args.save_results:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create serializable results
        serializable_results = {}
        for domain, result in all_results.items():
            serializable_results[domain] = {
                "total": result["total"],
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "results": [
                    {
                        "prompt": r["prompt"],
                        "expected": r.get("expected"),  # Handle None case
                        "tool_call": _to_serializable(r["tool_call"]),
                        "baseline": _to_serializable(r["baseline"]),
                        "evaluation": r["evaluation"],
                    }
                    for r in result["results"]
                ]
            }
        
        # Save with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"benchmark_results_{timestamp}.json"
        
        serializable_metrics = {}
        for domain, metric in metrics.items():
            serializable_metrics[domain] = {
                "domain": metric.domain,
                "total_queries": metric.total_queries,
                "domain_detection_accuracy": metric.domain_detection_accuracy,
                "tool_success_rate": metric.tool_success_rate,
                "answer_accuracy": metric.answer_accuracy,
                "baseline_accuracy": metric.baseline_accuracy,
                "avg_tool_time_ms": metric.avg_tool_time_ms,
                "avg_baseline_time_ms": metric.avg_baseline_time_ms,
                "improvement_ratio": metric.improvement_ratio,
            }

        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "model_path": wrapper.model_path,
                "dataset": args.dataset,
                "mlx_available": wrapper.get_model_info()["mlx_available"],
                "total_domains": len(all_results),
                "domains_tested": test_domains,
                "domain_metrics": serializable_metrics,
                "results": serializable_results,
            }, f, indent=2)
        
        if evaluator.verbose:
            print(f"\nResults saved to: {output_file}")
    
    return {
        "total_queries": total_queries,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "overall_baseline_accuracy": overall_baseline_accuracy,
        "domain_results": all_results,
    }


def run_quick_test(
    wrapper: Qwen3MLXWrapper,
    evaluator: Evaluator,
    args,
    baseline_runner: Optional[BaselineRunner],
    benchmark_prompts: Dict[str, Any],
):
    """Run quick test on 3 domains."""
    print_banner()
    
    # Test arithmetic, physics, and math (diverse set)
    test_domains = ["arithmetic", "physics", "math"]
    
    print(f"Quick test on domains: {', '.join(test_domains)}")
    print()
    
    all_results = {}
    for domain in test_domains:
        domain_result = run_domain_benchmark(
            wrapper=wrapper,
            evaluator=evaluator,
            domain=domain,
            benchmark_prompts=benchmark_prompts,
            num_samples=1,  # Just 1 quick test per domain
            baseline_runner=baseline_runner,
        )
        
        all_results[domain] = domain_result
    
    # Quick summary
    print("\nQuick Test Summary:")
    for domain, result in all_results.items():
        status = "✓" if result["correct"] else "✗"
        first_result = result["results"][0] if result["results"] else {}
        tool_response = first_result.get("tool_call", {}).get("response", "N/A")
        print(f"  {domain.upper()}: {status} ({str(tool_response)[:50]})")
    
    return all_results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create wrapper and evaluator
    wrapper = create_wrapper(
        model_path=args.model_path,
        use_thinking=True,
        temperature=0.6,
        max_tokens=2048,
        tool_selection=args.tool_selection,
        llm_query_extraction=args.llm_query_extraction,
        verbose=args.verbose,
    )
    evaluator = create_evaluator(verbose=args.verbose)
    
    # Load model
    if not wrapper.load_model():
        print("Warning: Failed to load model. Proceeding with fallback mode.")
        print("To use MLX models, set FLUXEM_ENABLE_MLX=1 and update --model-path.")

    baseline_runner = _create_baseline_runner(args, wrapper)
    benchmark_prompts = _load_benchmark_prompts(args.dataset)
    
    # Run benchmark
    if args.quick_test:
        results = run_quick_test(wrapper, evaluator, args, baseline_runner, benchmark_prompts)
    else:
        results = run_full_benchmark(wrapper, evaluator, args, baseline_runner, benchmark_prompts)
    
    # Get and print full report
    if not args.quick_test:
        metrics = evaluator.aggregate_metrics()
        report = evaluator.generate_report(metrics)
        print(report)
    
    # Save results for quick test runs
    if args.save_results and args.quick_test:
        evaluator.save_results(args.output)


if __name__ == "__main__":
    main()
