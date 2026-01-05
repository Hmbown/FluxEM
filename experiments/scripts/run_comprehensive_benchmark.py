#!/usr/bin/env python3
"""
Comprehensive benchmark runner for FluxEM.

Executes comparison scripts and emits data-only output plus a JSON results file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def run_script_as_module(script_path: Path) -> Tuple[bool, str, Optional[str]]:
    """
    Run a script by importing it as a module and calling its main() function.

    Returns: (success, output, error_message)
    """
    old_argv = sys.argv
    sys.argv = [str(script_path)]
    captured_output = StringIO()

    try:
        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        if spec is None or spec.loader is None:
            return False, "", f"Could not load spec for {script_path}"

        module = importlib.util.module_from_spec(spec)
        sys.modules[script_path.stem] = module

        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            spec.loader.exec_module(module)
            if hasattr(module, "main"):
                module.main()
            output = captured_output.getvalue()
            return True, output, None
        finally:
            sys.stdout = old_stdout
            if script_path.stem in sys.modules:
                del sys.modules[script_path.stem]

    except SystemExit as e:
        output = captured_output.getvalue()
        if e.code == 0 or e.code is None:
            return True, output, None
        return False, output, f"Script exited with code {e.code}"

    except Exception as e:
        output = captured_output.getvalue()
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return False, output, error_msg

    finally:
        sys.argv = old_argv


def extract_metrics_from_output(output: str) -> Dict[str, Any]:
    """Extract simple metrics from script output."""
    metrics: Dict[str, Any] = {}
    import re

    table_row = re.search(r'FluxEM\s+([\d.]+)%', output)
    if table_row:
        metrics['fluxem_accuracy'] = float(table_row.group(1))

    fluxem_acc = re.findall(r'FluxEM[^:\d]*:\s*([\d.]+)%', output, re.IGNORECASE)
    if fluxem_acc and 'fluxem_accuracy' not in metrics:
        metrics['fluxem_accuracy'] = float(fluxem_acc[-1])

    accuracy = re.findall(r'Accuracy[^:]*:\s*([\d.]+)%', output, re.IGNORECASE)
    if accuracy:
        metrics['accuracy'] = float(accuracy[-1])

    errors = re.findall(r'(?:relative\s+)?error[^:]*:\s*([\d.e+-]+)', output, re.IGNORECASE)
    if errors:
        try:
            metrics['error'] = float(errors[-1])
        except ValueError:
            pass

    ood_acc = re.findall(r'OOD[^:]*:\s*([\d.]+)%', output, re.IGNORECASE)
    if ood_acc:
        metrics['ood_accuracy'] = float(ood_acc[-1])

    return metrics


def run_script(script_name: str, scripts_dir: Path) -> ScriptResult:
    """Run a single comparison script and return its result."""
    script_path = scripts_dir / script_name

    if not script_path.exists():
        return ScriptResult(
            script_name=script_name,
            success=False,
            duration_seconds=0.0,
            output="",
            error=f"Script not found: {script_path}",
        )

    start_time = time.time()
    success, output, error = run_script_as_module(script_path)
    duration = time.time() - start_time

    metrics = extract_metrics_from_output(output)

    return ScriptResult(
        script_name=script_name,
        success=success,
        duration_seconds=duration,
        output=output,
        error=error,
        metrics=metrics,
    )


def run_comprehensive_benchmark(
    scripts_to_run: Optional[List[str]] = None,
    scripts_to_skip: Optional[List[str]] = None,
) -> BenchmarkResults:
    """Run the comprehensive benchmark suite."""
    if scripts_to_run:
        normalized = [s if s.endswith('.py') else s + '.py' for s in scripts_to_run]
        scripts = [s for s in normalized if s in ALL_SCRIPTS]
    else:
        scripts = ALL_SCRIPTS.copy()

    skip_set = set(scripts_to_skip or [])
    skip_set = {s if s.endswith('.py') else s + '.py' for s in skip_set}
    scripts = [s for s in scripts if s not in skip_set]

    timestamp = datetime.now().isoformat()
    start_time = time.time()

    script_results: List[ScriptResult] = []
    skipped_count = len(ALL_SCRIPTS) - len(scripts) - len([s for s in skip_set if s in ALL_SCRIPTS])

    for script_name in scripts:
        result = run_script(script_name, SCRIPTS_DIR)
        script_results.append(result)

    total_duration = time.time() - start_time
    successful = sum(1 for r in script_results if r.success)
    failed = sum(1 for r in script_results if not r.success)

    summary_metrics: Dict[str, Any] = {}
    fluxem_accuracies = [
        r.metrics['fluxem_accuracy']
        for r in script_results
        if 'fluxem_accuracy' in r.metrics
    ]
    if fluxem_accuracies:
        summary_metrics['avg_fluxem_accuracy'] = sum(fluxem_accuracies) / len(fluxem_accuracies)
        summary_metrics['min_fluxem_accuracy'] = min(fluxem_accuracies)
        summary_metrics['max_fluxem_accuracy'] = max(fluxem_accuracies)

    return BenchmarkResults(
        timestamp=timestamp,
        total_scripts=len(scripts),
        successful=successful,
        failed=failed,
        skipped=skipped_count,
        total_duration_seconds=total_duration,
        script_results=script_results,
        summary_metrics=summary_metrics,
    )


def save_results(results: BenchmarkResults, output_path: Optional[Path]) -> Path:
    """Save results to JSON file and return the path."""
    if output_path:
        output_dir = output_path if output_path.is_dir() else output_path.parent
        base_name = output_path.stem if not output_path.is_dir() else "benchmark"
    else:
        output_dir = SCRIPTS_DIR.parent / "results"
        base_name = "benchmark"

    output_dir.mkdir(parents=True, exist_ok=True)
    ts_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{base_name}_results_{ts_suffix}.json"

    payload = {
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

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comprehensive FluxEM benchmark suite",
    )
    parser.add_argument("--skip", nargs="+", default=[], help="Scripts to skip")
    parser.add_argument("--only", nargs="+", default=None, help="Only run these scripts")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for results")
    parser.add_argument("--list", action="store_true", help="List scripts and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show scripts without executing")

    args = parser.parse_args()

    if args.list:
        print("table=available_scripts")
        print("script_name\texists")
        for script in ALL_SCRIPTS:
            exists = (SCRIPTS_DIR / script).exists()
            print(f"{script}\t{int(exists)}")
        return

    if args.dry_run:
        if args.only:
            scripts_to_run = [s if s.endswith('.py') else s + '.py' for s in args.only]
            scripts_to_run = [s for s in scripts_to_run if s in ALL_SCRIPTS]
        else:
            scripts_to_run = ALL_SCRIPTS.copy()

        scripts_to_skip = {s if s.endswith('.py') else s + '.py' for s in args.skip}
        scripts_to_run = [s for s in scripts_to_run if s not in scripts_to_skip]

        print("table=dry_run")
        print("script_name\texists")
        for script in scripts_to_run:
            exists = (SCRIPTS_DIR / script).exists()
            print(f"{script}\t{int(exists)}")
        return

    output_path = Path(args.output) if args.output else None
    results = run_comprehensive_benchmark(
        scripts_to_run=args.only,
        scripts_to_skip=args.skip,
    )

    print("table=run_context")
    print("field\tvalue")
    print(f"timestamp\t{results.timestamp}")
    print(f"total_scripts\t{results.total_scripts}")
    print(f"successful\t{results.successful}")
    print(f"failed\t{results.failed}")
    print(f"skipped\t{results.skipped}")
    print(f"total_duration_seconds\t{results.total_duration_seconds:.2f}")

    print("table=script_results")
    print("script_name\tsuccess\tduration_seconds\terror")
    for r in results.script_results:
        error = r.error.replace("\n", " ") if r.error else ""
        print(f"{r.script_name}\t{int(r.success)}\t{r.duration_seconds:.2f}\t{error}")

    if results.summary_metrics:
        print("table=summary_metrics")
        print("metric\tvalue")
        for key, value in results.summary_metrics.items():
            if isinstance(value, float):
                print(f"{key}\t{value:.6f}")
            else:
                print(f"{key}\t{value}")

    json_path = save_results(results, output_path)
    print(f"results_path\t{json_path}")

    if results.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
