#!/usr/bin/env python3
"""
Visualize FluxEM experiment results.

Loads result JSON files and emits a TSV table of metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def find_result_files(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Find and load result files in the results directory."""
    results: Dict[str, Dict[str, Any]] = {}

    for eval_file in results_dir.glob("**/evaluation_results.json"):
        exp_name = eval_file.parent.name
        try:
            results[exp_name] = load_json(eval_file)
        except Exception as e:
            print(f"warning\tload_failed\tfile={eval_file}\terror={e}")

    for comp_file in results_dir.glob("**/embedding_comparison.json"):
        try:
            data = load_json(comp_file)
            if "results" in data:
                results["embedding_comparison"] = data["results"]
        except Exception as e:
            print(f"warning\tload_failed\tfile={comp_file}\terror={e}")

    for json_file in results_dir.glob("*.json"):
        if json_file.name in ["evaluation_results.json", "embedding_comparison.json"]:
            continue
        if "training_history" in json_file.name:
            continue
        try:
            exp_name = json_file.stem
            results[exp_name] = load_json(json_file)
        except Exception as e:
            print(f"warning\tload_failed\tfile={json_file}\terror={e}")

    return results


def extract_metrics(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Extract metrics into a standardized format."""
    extracted: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for exp_name, exp_results in results.items():
        if "_training" in exp_name:
            continue
        if not isinstance(exp_results, dict):
            continue

        extracted[exp_name] = {}
        for method, method_results in exp_results.items():
            if isinstance(method_results, dict):
                extracted[exp_name][method] = method_results

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FluxEM experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for TSV metrics",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"error\tresults_dir_not_found\tpath={results_dir}")
        return

    all_results = find_result_files(results_dir)
    if not all_results:
        print("error\tno_results_found")
        return

    metrics = extract_metrics(all_results)

    lines = []
    lines.append("table=metrics")
    lines.append("experiment\tmethod\tsplit\tmetric\tvalue")

    for exp_name, exp_data in metrics.items():
        for method, method_results in exp_data.items():
            if not isinstance(method_results, dict):
                continue
            for split, split_metrics in method_results.items():
                if not isinstance(split_metrics, dict):
                    continue
                for metric_name, value in split_metrics.items():
                    lines.append(f"{exp_name}\t{method}\t{split}\t{metric_name}\t{value}")

    output_text = "\n".join(lines)
    print(output_text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
        print(f"results_path\t{output_path}")


if __name__ == "__main__":
    main()
