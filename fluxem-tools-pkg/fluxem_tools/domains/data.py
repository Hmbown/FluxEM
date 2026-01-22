"""Data domain - arrays, records, tables.

This module provides deterministic data structure operations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def array_summary(arr: List) -> Dict[str, Any]:
    """Summarize a numeric array.

    Returns:
        Dict with length, min, max, mean, sum, and flags
    """
    if not arr:
        return {"length": 0, "empty": True}

    numeric = [x for x in arr if isinstance(x, (int, float))]
    if not numeric:
        return {"length": len(arr), "numeric_count": 0}

    return {
        "length": len(arr),
        "numeric_count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": sum(numeric) / len(numeric),
        "sum": sum(numeric),
        "has_negatives": any(x < 0 for x in numeric),
        "all_integers": all(isinstance(x, int) or x == int(x) for x in numeric),
    }


def array_length(arr: List) -> int:
    """Return the length of an array."""
    return len(arr)


def array_sort(arr: List) -> List:
    """Sort an array in ascending order."""
    return sorted(arr)


def array_unique(arr: List) -> List:
    """Return unique values from an array, sorted."""
    try:
        return sorted(set(arr))
    except TypeError:
        # Handle unhashable types
        seen = []
        for item in arr:
            if item not in seen:
                seen.append(item)
        return seen


def array_reverse(arr: List) -> List:
    """Reverse an array."""
    return list(reversed(arr))


def array_slice(arr: List, start: int, end: int = None) -> List:
    """Slice an array from start to end (exclusive)."""
    if end is None:
        return arr[start:]
    return arr[start:end]


def record_schema(record: Dict) -> Dict[str, Any]:
    """Summarize a record's schema.

    Returns:
        Dict with field count, field names, and types
    """
    return {
        "num_fields": len(record),
        "fields": list(record.keys()),
        "types": {k: type(v).__name__ for k, v in record.items()},
    }


def table_summary(table: Union[List[Dict], Dict[str, List]]) -> Dict[str, Any]:
    """Summarize a table (list of records or column dict).

    Returns:
        Dict with rows, columns, and column types
    """
    if isinstance(table, dict):
        # Column-oriented
        cols = list(table.keys())
        n_rows = len(table[cols[0]]) if cols else 0
        return {
            "n_rows": n_rows,
            "n_cols": len(cols),
            "columns": cols,
        }
    elif isinstance(table, list) and table and isinstance(table[0], dict):
        # Row-oriented
        cols = list(table[0].keys()) if table else []
        return {
            "n_rows": len(table),
            "n_cols": len(cols),
            "columns": cols,
        }
    return {"error": "Invalid table format"}


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_array(args) -> List:
    if isinstance(args, dict):
        arr = args.get("array", args.get("arr", args.get("data", args.get("values"))))
        if arr is not None:
            return list(arr)
        return list(list(args.values())[0])
    if isinstance(args, (list, tuple)):
        return list(args)
    raise ValueError(f"Cannot parse array: {args}")


def _parse_record(args) -> Dict:
    if isinstance(args, dict):
        rec = args.get("record", args.get("data"))
        if rec is not None and isinstance(rec, dict):
            return rec
        return args
    raise ValueError(f"Cannot parse record: {args}")


def _parse_table(args) -> Union[List[Dict], Dict[str, List]]:
    if isinstance(args, dict):
        table = args.get("table", args.get("data"))
        if table is not None:
            return table
        return args
    if isinstance(args, list):
        return args
    raise ValueError(f"Cannot parse table: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register data tools in the registry."""

    registry.register(ToolSpec(
        name="data_array_summary",
        function=lambda args: array_summary(_parse_array(args)),
        description="Summarizes a numeric array (length, min, max, mean, sum, flags).",
        parameters={
            "type": "object",
            "properties": {
                "array": {"type": "array", "description": "Array to summarize"}
            },
            "required": ["array"]
        },
        returns="Summary dict with statistics",
        examples=[
            {"input": {"array": [1, 2, 3, 4, 5]}, "output": {"length": 5, "min": 1, "max": 5, "mean": 3.0, "sum": 15}},
        ],
        domain="data",
        tags=["array", "summary", "statistics"],
    ))

    registry.register(ToolSpec(
        name="data_array_length",
        function=lambda args: array_length(_parse_array(args)),
        description="Returns the length of an array.",
        parameters={
            "type": "object",
            "properties": {
                "array": {"type": "array", "description": "Array to measure"}
            },
            "required": ["array"]
        },
        returns="Length as integer",
        examples=[
            {"input": {"array": [1, 2, 3]}, "output": 3},
        ],
        domain="data",
        tags=["array", "length", "count"],
    ))

    registry.register(ToolSpec(
        name="data_record_schema",
        function=lambda args: record_schema(_parse_record(args)),
        description="Summarizes a record's schema (field count, names, types).",
        parameters={
            "type": "object",
            "properties": {
                "record": {"type": "object", "description": "Record/object to analyze"}
            },
            "required": ["record"]
        },
        returns="Schema dict",
        examples=[
            {"input": {"record": {"name": "Ada", "age": 37}}, "output": {"num_fields": 2, "fields": ["name", "age"]}},
        ],
        domain="data",
        tags=["record", "schema", "structure"],
    ))

    registry.register(ToolSpec(
        name="data_table_summary",
        function=lambda args: table_summary(_parse_table(args)),
        description="Summarizes a table (rows, columns, column names).",
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "array", "description": "Table as list of records or column dict"}
            },
            "required": ["table"]
        },
        returns="Table summary dict",
        examples=[
            {"input": {"table": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}, "output": {"n_rows": 2, "n_cols": 2, "columns": ["a", "b"]}},
        ],
        domain="data",
        tags=["table", "summary", "structure"],
    ))

    registry.register(ToolSpec(
        name="data_array_sort",
        function=lambda args: array_sort(_parse_array(args)),
        description="Sorts an array in ascending order.",
        parameters={
            "type": "object",
            "properties": {
                "array": {"type": "array", "description": "Array to sort"}
            },
            "required": ["array"]
        },
        returns="Sorted array",
        examples=[
            {"input": {"array": [3, 1, 4, 1, 5]}, "output": [1, 1, 3, 4, 5]},
        ],
        domain="data",
        tags=["array", "sort", "order"],
    ))

    registry.register(ToolSpec(
        name="data_unique",
        function=lambda args: array_unique(_parse_array(args)),
        description="Returns unique values from an array, sorted.",
        parameters={
            "type": "object",
            "properties": {
                "array": {"type": "array", "description": "Array to deduplicate"}
            },
            "required": ["array"]
        },
        returns="Array of unique values",
        examples=[
            {"input": {"array": [1, 2, 2, 3, 3, 3]}, "output": [1, 2, 3]},
        ],
        domain="data",
        tags=["array", "unique", "distinct"],
    ))
