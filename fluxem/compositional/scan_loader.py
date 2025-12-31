"""
SCAN dataset loader.

SCAN format: IN: <command> OUT: <actions>
Each line is one example with input command and expected output actions.

To use, download SCAN from: https://github.com/brendenlake/SCAN
Then provide the path to load_scan_split().
"""

from pathlib import Path
from typing import List, Tuple, Optional


def load_scan_file(filepath: Path | str) -> List[Tuple[str, str]]:
    """
    Load a SCAN dataset file.

    Args:
        filepath: Path to SCAN file

    Returns:
        List of (command, actions) tuples
    """
    filepath = Path(filepath)
    examples = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse "IN: <command> OUT: <actions>"
            if line.startswith("IN:") and " OUT: " in line:
                parts = line.split(" OUT: ")
                command = parts[0].replace("IN: ", "").strip()
                actions = parts[1].strip()
                examples.append((command, actions))

    return examples


def load_scan_split(
    split_name: str,
    subset: str = "test",
    scan_dir: Optional[Path | str] = None
) -> List[Tuple[str, str]]:
    """
    Load a SCAN benchmark split.

    Args:
        split_name: One of "addprim_jump", "addprim_turn_left", "length", "simple"
        subset: "train" or "test"
        scan_dir: Path to SCAN directory (required)

    Returns:
        List of (command, actions) tuples

    Example:
        >>> data = load_scan_split("addprim_jump", "test", "/path/to/SCAN")
        >>> len(data)
        7706
    """
    if scan_dir is None:
        raise ValueError(
            "scan_dir is required. Download SCAN from "
            "https://github.com/brendenlake/SCAN and provide the path."
        )
    scan_dir = Path(scan_dir)

    # Map split names to directories and files
    split_map = {
        "addprim_jump": ("add_prim_split", f"tasks_{subset}_addprim_jump.txt"),
        "addprim_turn_left": ("add_prim_split", f"tasks_{subset}_addprim_turn_left.txt"),
        "length": ("length_split", f"tasks_{subset}_length.txt"),
        "simple": ("simple_split", f"tasks_{subset}_simple.txt"),
    }

    if split_name not in split_map:
        raise ValueError(f"Unknown split: {split_name}. Valid: {list(split_map.keys())}")

    subdir, filename = split_map[split_name]
    filepath = scan_dir / subdir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"SCAN file not found: {filepath}")

    return load_scan_file(filepath)


def get_split_stats(split_name: str, scan_dir: Path | str) -> dict:
    """Get statistics for a SCAN split."""
    train = load_scan_split(split_name, "train", scan_dir)
    test = load_scan_split(split_name, "test", scan_dir)

    return {
        "split": split_name,
        "train_size": len(train),
        "test_size": len(test),
        "train_example": train[0] if train else None,
        "test_example": test[0] if test else None,
    }
