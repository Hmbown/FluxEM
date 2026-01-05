#!/usr/bin/env python3
"""
Generate training and evaluation datasets for FluxEM experiments.

Supports:
- Arithmetic: expressions with ID/OOD splits by magnitude and length
- Units: dimensional analysis with type-checking tasks
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_arithmetic_sample(
    ops: List[str],
    num_range: Tuple[int, int],
    num_operands: int = 2,
) -> Dict[str, Any]:
    """Generate a single arithmetic sample."""
    operands = [random.randint(num_range[0], num_range[1]) for _ in range(num_operands)]
    
    # Build expression
    expr_parts = [str(operands[0])]
    current_result = float(operands[0])
    
    for i in range(1, num_operands):
        op = random.choice(ops)
        operand = operands[i]
        
        # Avoid division by zero
        if op == "/" and operand == 0:
            operand = 1
            operands[i] = 1
        
        expr_parts.append(op)
        expr_parts.append(str(operand))
        
        # Compute result
        if op == "+":
            current_result += operand
        elif op == "-":
            current_result -= operand
        elif op == "*":
            current_result *= operand
        elif op == "/":
            current_result /= operand
    
    text = " ".join(expr_parts)
    
    # Create spans for numeric values
    spans = []
    pos = 0
    for i, part in enumerate(expr_parts):
        if part not in ["+", "-", "*", "/"]:
            spans.append({
                "type": "arithmetic",
                "start": pos,
                "end": pos + len(part),
                "value": float(part),
            })
        pos += len(part) + 1  # +1 for space
    
    return {
        "text": text,
        "spans": spans,
        "target_text": str(current_result),
        "target_value": current_result,
    }


def generate_units_sample(
    unit_types: List[str],
    magnitude_range: Tuple[float, float],
    task_type: str = "conversion",
) -> Dict[str, Any]:
    """Generate a single units/dimensional sample."""
    
    # Unit definitions with dimensions
    units = {
        "length": {
            "m": {"L": 1}, "km": {"L": 1}, "cm": {"L": 1}, "mm": {"L": 1},
            "ft": {"L": 1}, "in": {"L": 1}, "mi": {"L": 1},
        },
        "mass": {
            "kg": {"M": 1}, "g": {"M": 1}, "mg": {"M": 1},
            "lb": {"M": 1}, "oz": {"M": 1},
        },
        "time": {
            "s": {"T": 1}, "ms": {"T": 1}, "min": {"T": 1},
            "h": {"T": 1}, "day": {"T": 1},
        },
        "temperature": {
            "K": {"Theta": 1}, "C": {"Theta": 1}, "F": {"Theta": 1},
        },
        "velocity": {
            "m/s": {"L": 1, "T": -1}, "km/h": {"L": 1, "T": -1},
            "mph": {"L": 1, "T": -1}, "ft/s": {"L": 1, "T": -1},
        },
    }
    
    if task_type == "can_add":
        # Generate two quantities and check if they can be added
        type1 = random.choice(unit_types)
        type2 = random.choice(unit_types)
        
        unit1 = random.choice(list(units[type1].keys()))
        unit2 = random.choice(list(units[type2].keys()))
        
        val1 = random.uniform(magnitude_range[0], magnitude_range[1])
        val2 = random.uniform(magnitude_range[0], magnitude_range[1])
        
        # Check if dimensions match
        dims1 = units[type1][unit1]
        dims2 = units[type2][unit2]
        can_add = dims1 == dims2
        
        text = f"Can you add {val1:.2f} {unit1} and {val2:.2f} {unit2}?"
        target = "true" if can_add else "false"
        
        spans = [
            {"type": "phys_quantity", "start": text.find(f"{val1:.2f}"),
             "end": text.find(unit1) + len(unit1), "value": val1, "dims": dims1},
            {"type": "phys_quantity", "start": text.find(f"{val2:.2f}", len(f"{val1:.2f}")),
             "end": text.rfind(unit2) + len(unit2), "value": val2, "dims": dims2},
        ]
        
        return {
            "text": text,
            "spans": spans,
            "target_text": target,
            "target_value": can_add,
        }
    
    else:  # conversion
        unit_type = random.choice(unit_types)
        unit_list = list(units[unit_type].keys())
        
        if len(unit_list) < 2:
            unit_list = unit_list * 2
        
        from_unit, to_unit = random.sample(unit_list, 2)
        value = random.uniform(magnitude_range[0], magnitude_range[1])
        
        # Simplified conversion (not physically accurate, but preserves structure)
        # In real use, would have proper conversion factors
        converted = value  # Placeholder
        
        text = f"Convert {value:.2f} {from_unit} to {to_unit}"
        dims = units[unit_type][from_unit]
        
        spans = [
            {"type": "phys_quantity", "start": text.find(f"{value:.2f}"),
             "end": text.find(from_unit) + len(from_unit), "value": value, "dims": dims},
        ]
        
        return {
            "text": text,
            "spans": spans,
            "target_text": f"{converted:.2f} {to_unit}",
            "target_value": converted,
        }


def generate_arithmetic_dataset(config: Dict) -> Dict[str, List[Dict]]:
    """Generate full arithmetic dataset with ID/OOD splits."""
    data_cfg = config["data"]
    ops = data_cfg.get("operations", ["+", "-", "*"])
    id_range = tuple(data_cfg.get("id_range", [0, 999]))
    ood_range = tuple(data_cfg.get("ood_magnitude_range", [1000000, 999999999]))
    ood_max_ops = data_cfg.get("ood_max_ops", 5)
    
    datasets = {}
    
    # Training data (ID)
    datasets["train"] = [
        generate_arithmetic_sample(ops, id_range, num_operands=2)
        for _ in range(data_cfg["train_size"])
    ]
    
    # ID test
    datasets["test_id"] = [
        generate_arithmetic_sample(ops, id_range, num_operands=2)
        for _ in range(data_cfg["test_size"])
    ]
    
    # OOD-A: Large magnitude
    datasets["test_ood_magnitude"] = [
        generate_arithmetic_sample(ops, ood_range, num_operands=2)
        for _ in range(data_cfg["test_size"])
    ]
    
    # OOD-B: Long chains
    datasets["test_ood_length"] = [
        generate_arithmetic_sample(ops, id_range, num_operands=random.randint(3, ood_max_ops))
        for _ in range(data_cfg["test_size"])
    ]
    
    return datasets


def generate_units_dataset(config: Dict) -> Dict[str, List[Dict]]:
    """Generate full units dataset with ID/OOD splits."""
    data_cfg = config["data"]
    unit_types = data_cfg.get("unit_types", ["length", "mass", "time"])
    id_range = tuple(data_cfg.get("id_magnitude_range", [0.001, 1000]))
    ood_range = tuple(data_cfg.get("ood_magnitude_range", [0.000001, 1000000000]))
    include_can_add = data_cfg.get("include_can_add", True)
    
    datasets = {}
    
    def gen_samples(n: int, mag_range: Tuple[float, float]) -> List[Dict]:
        samples = []
        for _ in range(n):
            if include_can_add and random.random() < 0.3:
                samples.append(generate_units_sample(unit_types, mag_range, "can_add"))
            else:
                samples.append(generate_units_sample(unit_types, mag_range, "conversion"))
        return samples
    
    # Training data
    datasets["train"] = gen_samples(data_cfg["train_size"], id_range)
    
    # ID test
    datasets["test_id"] = gen_samples(data_cfg["test_size"], id_range)
    
    # OOD: Extreme magnitudes
    datasets["test_ood_magnitude"] = gen_samples(data_cfg["test_size"], ood_range)
    
    return datasets


def save_jsonl(data: List[Dict], path: Path):
    """Save data as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(data)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate FluxEM experiment data")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set seed
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    random.seed(seed)
    print(f"Using seed: {seed}")
    
    # Generate based on task type
    task = config.get("task", "arithmetic")
    
    if task == "arithmetic":
        datasets = generate_arithmetic_dataset(config)
    elif task == "units":
        datasets = generate_units_dataset(config)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Save datasets
    data_dir = Path(config["paths"]["data_dir"])
    
    for name, data in datasets.items():
        save_jsonl(data, data_dir / f"{name}.jsonl")
    
    print(f"\nGenerated {task} dataset in {data_dir}")
    print(f"  Train: {len(datasets['train'])} samples")
    print(f"  Test splits: {[k for k in datasets.keys() if k != 'train']}")


if __name__ == "__main__":
    main()
