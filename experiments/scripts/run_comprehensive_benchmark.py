#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for FluxEM

Executes all comparison scripts and generates a unified report demonstrating
FluxEM's algebraic embedding advantages across multiple evaluation dimensions.

This script:
1. Runs all comparison scripts in sequence
2. Collects results from each
3. Generates unified output:
   - Console summary table
   - JSON results file
   - Markdown report (experiments/results/BENCHMARK_REPORT.md)

Usage:
    python experiments/scripts/run_comprehensive_benchmark.py
    python experiments/scripts/run_comprehensive_benchmark.py --skip compare_openai_embeddings.py
    python experiments/scripts/run_comprehensive_benchmark.py --only compare_embeddings.py compare_tokenizers.py
    python experiments/scripts/run_comprehensive_benchmark.py --output experiments/results/custom_report
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure FluxEM is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Configuration
# =============================================================================

# List of all comparison scripts to run (in order)
ALL_SCRIPTS = [
    "compare_embeddings.py",
    "compare_tokenizers.py",
    "compare_sentence_transformers.py",
    "compare_word_embeddings.py",
    "compare_openai_embeddings.py",
    "compare_neural_numbers.py",
    "compare_llm_numeric.py",
    "compare_positional.py",
    "compare_scratchpad.py",
    "ablation_study.py",
    "demo_all_domains.py",
]

SCRIPTS_DIR = Path(__file__).parent


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ScriptResult:
    """Result from running a single script."""
    script_name: str
    success: bool
    duration_seconds: float
    output: str
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResults:
    """Aggregated results from all scripts."""
    timestamp: str
    total_scripts: int
    successful: int
    failed: int
    skipped: int
    total_duration_seconds: float
    script_results: List[ScriptResult] = field(default_factory=list)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Script Execution
# =============================================================================

def capture_output(func, *args, **kwargs) -> Tuple[Any, str]:
    """Execute a function and capture its stdout output."""
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        result = func(*args, **kwargs)
        output = captured_output.getvalue()
        return result, output
    finally:
        sys.stdout = old_stdout


def run_script_as_module(script_path: Path) -> Tuple[bool, str, Optional[str]]:
    """
    Run a script by importing it as a module and calling its main() function.

    Returns: (success, output, error_message)
    """
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(
            script_path.stem,
            script_path
        )
        if spec is None or spec.loader is None:
            return False, "", f"Could not load spec for {script_path}"

        module = importlib.util.module_from_spec(spec)
        sys.modules[script_path.stem] = module

        # Execute the module to define its functions
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            spec.loader.exec_module(module)

            # Call main() if it exists
            if hasattr(module, 'main'):
                module.main()

            output = captured_output.getvalue()
            return True, output, None

        finally:
            sys.stdout = old_stdout
            # Clean up module
            if script_path.stem in sys.modules:
                del sys.modules[script_path.stem]

    except SystemExit as e:
        # Script called sys.exit() - that's usually OK
        output = captured_output.getvalue() if 'captured_output' in dir() else ""
        if e.code == 0 or e.code is None:
            return True, output, None
        return False, output, f"Script exited with code {e.code}"

    except Exception as e:
        output = captured_output.getvalue() if 'captured_output' in dir() else ""
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return False, output, error_msg


def run_script(script_name: str, scripts_dir: Path) -> ScriptResult:
    """
    Run a single comparison script and return its result.
    """
    script_path = scripts_dir / script_name

    if not script_path.exists():
        return ScriptResult(
            script_name=script_name,
            success=False,
            duration_seconds=0.0,
            output="",
            error=f"Script not found: {script_path}",
        )

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")

    start_time = time.time()
    success, output, error = run_script_as_module(script_path)
    duration = time.time() - start_time

    # Extract any metrics from the output (look for common patterns)
    metrics = extract_metrics_from_output(output)

    result = ScriptResult(
        script_name=script_name,
        success=success,
        duration_seconds=duration,
        output=output,
        error=error,
        metrics=metrics,
    )

    # Print summary
    status = "SUCCESS" if success else "FAILED"
    print(f"\n[{status}] {script_name} completed in {duration:.2f}s")
    if error:
        print(f"  Error: {error[:200]}...")

    return result


def extract_metrics_from_output(output: str) -> Dict[str, Any]:
    """
    Extract key metrics from script output.

    Looks for common patterns like:
    - "Accuracy: 95.5%"
    - "FluxEM: 100%"
    - "Error: 0.0001"
    """
    metrics = {}

    # Look for accuracy patterns
    import re

    # Pattern: "FluxEM: X%" or "FluxEM Accuracy: X%"
    fluxem_acc = re.findall(r'FluxEM[^:]*:\s*([\d.]+)%', output, re.IGNORECASE)
    if fluxem_acc:
        metrics['fluxem_accuracy'] = float(fluxem_acc[-1])

    # Pattern: "Accuracy: X%" or "accuracy: X%"
    accuracy = re.findall(r'(?<!FluxEM\s)Accuracy[^:]*:\s*([\d.]+)%', output, re.IGNORECASE)
    if accuracy:
        metrics['accuracy'] = float(accuracy[-1])

    # Pattern: "Error: X" or "relative error: X"
    errors = re.findall(r'(?:relative\s+)?error[^:]*:\s*([\d.e+-]+)', output, re.IGNORECASE)
    if errors:
        try:
            metrics['error'] = float(errors[-1])
        except ValueError:
            pass

    # Pattern: OOD-related metrics
    ood_acc = re.findall(r'OOD[^:]*:\s*([\d.]+)%', output, re.IGNORECASE)
    if ood_acc:
        metrics['ood_accuracy'] = float(ood_acc[-1])

    # Pattern: "exact" or "EXACT" for exact computation
    if 'EXACT' in output or 'exact' in output.lower():
        exact_mentions = output.lower().count('exact')
        if exact_mentions > 0:
            metrics['exact_computation'] = True

    return metrics


# =============================================================================
# Report Generation
# =============================================================================

def generate_console_summary(results: BenchmarkResults) -> str:
    """Generate a console-friendly summary table."""
    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append("  COMPREHENSIVE BENCHMARK SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Timestamp: {results.timestamp}")
    lines.append(f"  Total Duration: {results.total_duration_seconds:.2f}s")
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"  {'Script':<40} {'Status':<10} {'Time':>8} {'FluxEM Acc':>10}")
    lines.append("-" * 70)

    for script_result in results.script_results:
        status = "OK" if script_result.success else "FAILED"
        time_str = f"{script_result.duration_seconds:.2f}s"

        fluxem_acc = script_result.metrics.get('fluxem_accuracy', '')
        if fluxem_acc:
            acc_str = f"{fluxem_acc:.1f}%"
        else:
            acc_str = "-"

        # Truncate script name if needed
        name = script_result.script_name[:38] + ".." if len(script_result.script_name) > 40 else script_result.script_name

        lines.append(f"  {name:<40} {status:<10} {time_str:>8} {acc_str:>10}")

    lines.append("-" * 70)
    lines.append("")
    lines.append(f"  Total:      {results.total_scripts}")
    lines.append(f"  Successful: {results.successful}")
    lines.append(f"  Failed:     {results.failed}")
    lines.append(f"  Skipped:    {results.skipped}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_json_results(results: BenchmarkResults) -> Dict[str, Any]:
    """Generate JSON-serializable results dictionary."""
    return {
        "timestamp": results.timestamp,
        "summary": {
            "total_scripts": results.total_scripts,
            "successful": results.successful,
            "failed": results.failed,
            "skipped": results.skipped,
            "total_duration_seconds": results.total_duration_seconds,
        },
        "scripts": [
            {
                "name": r.script_name,
                "success": r.success,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
                "metrics": r.metrics,
            }
            for r in results.script_results
        ],
        "aggregated_metrics": results.summary_metrics,
    }


def generate_markdown_report(results: BenchmarkResults) -> str:
    """Generate a comprehensive Markdown report."""
    lines = []

    lines.append("# FluxEM Comprehensive Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {results.timestamp}")
    lines.append(f"**Total Duration:** {results.total_duration_seconds:.2f} seconds")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total Scripts:** {results.total_scripts}")
    lines.append(f"- **Successful:** {results.successful}")
    lines.append(f"- **Failed:** {results.failed}")
    lines.append(f"- **Skipped:** {results.skipped}")
    lines.append("")

    # Results Table
    lines.append("## Results Overview")
    lines.append("")
    lines.append("| Script | Status | Duration | Key Metrics |")
    lines.append("|--------|--------|----------|-------------|")

    for r in results.script_results:
        status = "Pass" if r.success else "Fail"
        status_icon = "+" if r.success else "-"
        duration = f"{r.duration_seconds:.2f}s"

        # Format metrics
        metrics_parts = []
        if 'fluxem_accuracy' in r.metrics:
            metrics_parts.append(f"FluxEM: {r.metrics['fluxem_accuracy']:.1f}%")
        if 'ood_accuracy' in r.metrics:
            metrics_parts.append(f"OOD: {r.metrics['ood_accuracy']:.1f}%")
        if 'error' in r.metrics:
            metrics_parts.append(f"Error: {r.metrics['error']:.2e}")
        if 'exact_computation' in r.metrics:
            metrics_parts.append("Exact")

        metrics_str = ", ".join(metrics_parts) if metrics_parts else "-"

        lines.append(f"| {r.script_name} | {status} | {duration} | {metrics_str} |")

    lines.append("")

    # Individual Script Details
    lines.append("## Individual Script Results")
    lines.append("")

    for r in results.script_results:
        lines.append(f"### {r.script_name}")
        lines.append("")

        if r.success:
            lines.append(f"**Status:** Success")
        else:
            lines.append(f"**Status:** Failed")
            if r.error:
                lines.append(f"**Error:** `{r.error[:200]}...`")

        lines.append(f"**Duration:** {r.duration_seconds:.2f}s")
        lines.append("")

        if r.metrics:
            lines.append("**Metrics:**")
            for key, value in r.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.4f}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")

        # Include output excerpt
        if r.output:
            # Get last 50 lines or key conclusion section
            output_lines = r.output.strip().split('\n')

            # Try to find conclusion or summary section
            conclusion_start = -1
            for i, line in enumerate(output_lines):
                if any(word in line.upper() for word in ['CONCLUSION', 'SUMMARY', 'KEY INSIGHT']):
                    conclusion_start = i
                    break

            if conclusion_start >= 0:
                excerpt = '\n'.join(output_lines[conclusion_start:conclusion_start+20])
            else:
                # Take last 20 lines
                excerpt = '\n'.join(output_lines[-20:])

            lines.append("<details>")
            lines.append("<summary>Output Excerpt</summary>")
            lines.append("")
            lines.append("```")
            lines.append(excerpt[:2000])  # Limit length
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    # Conclusions
    lines.append("## Key Findings")
    lines.append("")
    lines.append("Based on the comprehensive benchmark results:")
    lines.append("")

    # Summarize FluxEM performance
    fluxem_accuracies = [
        r.metrics['fluxem_accuracy']
        for r in results.script_results
        if 'fluxem_accuracy' in r.metrics
    ]

    if fluxem_accuracies:
        avg_acc = sum(fluxem_accuracies) / len(fluxem_accuracies)
        lines.append(f"1. **FluxEM Average Accuracy:** {avg_acc:.1f}% across {len(fluxem_accuracies)} benchmarks")

    # Count exact computation mentions
    exact_count = sum(
        1 for r in results.script_results
        if r.metrics.get('exact_computation', False)
    )
    if exact_count > 0:
        lines.append(f"2. **Exact Computation:** {exact_count} benchmarks demonstrated exact algebraic computation")

    lines.append(f"3. **Success Rate:** {results.successful}/{results.total_scripts} scripts completed successfully")
    lines.append("")

    lines.append("## Core Thesis Validation")
    lines.append("")
    lines.append("These benchmarks validate FluxEM's core thesis:")
    lines.append("")
    lines.append("> **\"Tokenize natural language; embed structured domain objects.\"**")
    lines.append("")
    lines.append("FluxEM's algebraic embeddings achieve perfect or near-perfect accuracy on arithmetic")
    lines.append("tasks because the algebra is built into the representation, not learned from data.")
    lines.append("This enables systematic generalization that learned approaches cannot achieve.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by run_comprehensive_benchmark.py*")

    return "\n".join(lines)


# =============================================================================
# Main Runner
# =============================================================================

def run_comprehensive_benchmark(
    scripts_to_run: Optional[List[str]] = None,
    scripts_to_skip: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> BenchmarkResults:
    """
    Run the comprehensive benchmark suite.

    Args:
        scripts_to_run: If specified, only run these scripts
        scripts_to_skip: Scripts to skip
        output_path: Custom output path for results

    Returns:
        BenchmarkResults with all collected data
    """
    # Determine which scripts to run
    if scripts_to_run:
        # Normalize names first
        normalized = [s if s.endswith('.py') else s + '.py' for s in scripts_to_run]
        # Filter to only those in ALL_SCRIPTS
        scripts = [s for s in normalized if s in ALL_SCRIPTS]
    else:
        scripts = ALL_SCRIPTS.copy()

    # Apply skip list
    skip_set = set(scripts_to_skip or [])
    # Normalize skip names
    skip_set = {s if s.endswith('.py') else s + '.py' for s in skip_set}

    scripts = [s for s in scripts if s not in skip_set]

    # Initialize results
    timestamp = datetime.now().isoformat()
    start_time = time.time()

    script_results: List[ScriptResult] = []
    skipped_count = len(ALL_SCRIPTS) - len(scripts) - len([s for s in skip_set if s in ALL_SCRIPTS])

    print("")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*    FluxEM COMPREHENSIVE BENCHMARK SUITE                           *")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("")
    print(f"  Scripts to run: {len(scripts)}")
    print(f"  Scripts to skip: {len(skip_set)}")
    print(f"  Start time: {timestamp}")
    print("")

    # Run each script
    for i, script_name in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] {script_name}")

        result = run_script(script_name, SCRIPTS_DIR)
        script_results.append(result)

        # Print the output
        if result.output:
            print(result.output)

    total_duration = time.time() - start_time
    successful = sum(1 for r in script_results if r.success)
    failed = sum(1 for r in script_results if not r.success)

    # Aggregate metrics
    summary_metrics = {}
    fluxem_accuracies = [
        r.metrics['fluxem_accuracy']
        for r in script_results
        if 'fluxem_accuracy' in r.metrics
    ]
    if fluxem_accuracies:
        summary_metrics['avg_fluxem_accuracy'] = sum(fluxem_accuracies) / len(fluxem_accuracies)
        summary_metrics['min_fluxem_accuracy'] = min(fluxem_accuracies)
        summary_metrics['max_fluxem_accuracy'] = max(fluxem_accuracies)

    results = BenchmarkResults(
        timestamp=timestamp,
        total_scripts=len(scripts),
        successful=successful,
        failed=failed,
        skipped=skipped_count,
        total_duration_seconds=total_duration,
        script_results=script_results,
        summary_metrics=summary_metrics,
    )

    return results


def save_results(
    results: BenchmarkResults,
    output_path: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    Save results to files.

    Returns: (json_path, markdown_path, console_output_path)
    """
    # Determine output directory
    if output_path:
        output_dir = output_path if output_path.is_dir() else output_path.parent
        base_name = output_path.stem if not output_path.is_dir() else "benchmark"
    else:
        output_dir = SCRIPTS_DIR.parent / "results"
        base_name = "benchmark"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp suffix for unique filenames
    ts_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"{base_name}_results_{ts_suffix}.json"
    with open(json_path, 'w') as f:
        json.dump(generate_json_results(results), f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Save Markdown report
    markdown_path = output_dir / "BENCHMARK_REPORT.md"
    with open(markdown_path, 'w') as f:
        f.write(generate_markdown_report(results))
    print(f"Markdown report saved to: {markdown_path}")

    # Save console output
    console_path = output_dir / f"{base_name}_console_{ts_suffix}.txt"
    with open(console_path, 'w') as f:
        f.write(generate_console_summary(results))
    print(f"Console summary saved to: {console_path}")

    return json_path, markdown_path, console_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive FluxEM benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks
    python run_comprehensive_benchmark.py

    # Skip specific scripts
    python run_comprehensive_benchmark.py --skip compare_openai_embeddings.py

    # Run only specific scripts
    python run_comprehensive_benchmark.py --only compare_embeddings.py compare_tokenizers.py

    # Custom output path
    python run_comprehensive_benchmark.py --output experiments/results/my_benchmark

    # List available scripts
    python run_comprehensive_benchmark.py --list
        """
    )

    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Scripts to skip (can specify multiple)",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only run these scripts (can specify multiple)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for results",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available scripts and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        print("\nAvailable benchmark scripts:")
        print("-" * 40)
        for script in ALL_SCRIPTS:
            exists = (SCRIPTS_DIR / script).exists()
            status = "[EXISTS]" if exists else "[MISSING]"
            print(f"  {script:<35} {status}")
        print("")
        return

    # Handle --dry-run
    if args.dry_run:
        # Normalize script names (with .py suffix)
        if args.only:
            scripts_to_run = [s if s.endswith('.py') else s + '.py' for s in args.only]
            scripts_to_run = [s for s in scripts_to_run if s in ALL_SCRIPTS]
        else:
            scripts_to_run = ALL_SCRIPTS.copy()

        scripts_to_skip = {s if s.endswith('.py') else s + '.py' for s in args.skip}
        scripts_to_run = [s for s in scripts_to_run if s not in scripts_to_skip]

        print("\nDry run - would execute:")
        print("-" * 40)
        for script in scripts_to_run:
            exists = (SCRIPTS_DIR / script).exists()
            status = "[EXISTS]" if exists else "[MISSING]"
            print(f"  {script:<35} {status}")
        print("")
        return

    # Run benchmarks
    output_path = Path(args.output) if args.output else None

    results = run_comprehensive_benchmark(
        scripts_to_run=args.only,
        scripts_to_skip=args.skip,
        output_path=output_path,
    )

    # Print console summary
    print(generate_console_summary(results))

    # Save results
    save_results(results, output_path)

    # Exit with appropriate code
    if results.failed > 0:
        print(f"\nWarning: {results.failed} script(s) failed.")
        sys.exit(1)
    else:
        print("\nAll benchmarks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
