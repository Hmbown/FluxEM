#!/usr/bin/env python3
"""
Migration script to convert fluxem-domains files to use backend abstraction.
"""

import re
import os
import sys
from pathlib import Path


def migrate_file(content: str, depth: int = 3) -> str:
    """
    Migrate a file from MLX to backend abstraction.

    Args:
        content: Original file content
        depth: Depth of the module (for relative imports)

    Returns:
        Migrated file content
    """
    # Build the relative import prefix based on depth
    dots = "." * depth

    # FIRST: Handle type hints BEFORE replacing mx.* calls
    # Type hints: ": mx.array" or "-> mx.array" -> Any
    content = re.sub(r':\s*mx\.array\b', ': Any', content)
    content = re.sub(r'->\s*mx\.array\b', '-> Any', content)

    # Replace MLX import with backend import
    content = re.sub(
        r'import mlx\.core as mx\n',
        f'from {dots}backend import get_backend\n',
        content
    )

    # Update core.base import path
    content = re.sub(
        r'from \.\.core\.base import',
        f'from {dots}core.base import',
        content
    )

    # Update other relative imports (need to handle domains too)
    content = re.sub(
        r'from \.\.(\w+) import',
        lambda m: f'from {dots}{m.group(1)} import',
        content
    )

    # Replace mx.* calls with backend.* calls
    content = re.sub(r'\bmx\.array\b', 'backend.array', content)
    content = re.sub(r'\bmx\.zeros\b', 'backend.zeros', content)
    content = re.sub(r'\bmx\.ones\b', 'backend.ones', content)
    content = re.sub(r', dtype=mx\.float32\)', ')', content)  # Remove dtype
    content = re.sub(r', dtype=mx\.int32\)', ')', content)
    content = re.sub(r'dtype=mx\.float32', 'dtype=None', content)
    content = re.sub(r'dtype=mx\.int32', 'dtype=None', content)

    # Replace mx.* operations
    mx_ops = [
        'allclose', 'sum', 'mean', 'abs', 'sqrt', 'exp', 'log', 'log10',
        'sin', 'cos', 'tan', 'arctan2', 'floor', 'ceil', 'clip', 'sign',
        'power', 'maximum', 'minimum', 'dot', 'matmul', 'argmax', 'argmin',
        'where', 'concatenate', 'stack', 'reshape', 'arange', 'linspace',
        'eye', 'all', 'any', 'prod', 'squeeze', 'expand_dims', 'transpose'
    ]
    for op in mx_ops:
        content = re.sub(rf'\bmx\.{op}\b', f'backend.{op}', content)

    # Replace .at[...].add(...) pattern with backend.at_add(...)
    # Pattern: emb.at[expr].add(value) -> backend.at_add(emb, expr, value)
    # Use non-greedy matching and handle nested brackets carefully

    # Simple case: emb.at[i].add(value)
    def replace_at_add(match):
        var = match.group(1)
        idx = match.group(2)
        val = match.group(3)
        return f'backend.at_add({var}, {idx}, {val})'

    # Match patterns like: var.at[...].add(...)
    # Handle slice notation too
    content = re.sub(
        r'(\w+)\.at\[([^\]]+)\]\.add\(([^)]+)\)',
        replace_at_add,
        content
    )

    # Handle slice with colon: emb.at[0:8].add(x)
    content = re.sub(
        r'backend\.at_add\((\w+), (\d+):(\d+), ([^)]+)\)',
        r'backend.at_add(\1, slice(\2, \3), \4)',
        content
    )

    # Handle slice starting at 0: emb.at[:8] -> slice(0, 8)
    content = re.sub(
        r'backend\.at_add\((\w+), :(\d+), ([^)]+)\)',
        r'backend.at_add(\1, slice(0, \2), \3)',
        content
    )

    # Handle slice with expressions: emb.at[start:end] where start/end are expressions
    # Pattern: backend.at_add(var, expr1:expr2, val) -> backend.at_add(var, slice(expr1, expr2), val)
    def convert_expr_slice(match):
        var = match.group(1)
        expr1 = match.group(2)
        expr2 = match.group(3)
        val = match.group(4)
        return f'backend.at_add({var}, slice({expr1}, {expr2}), {val})'

    # Match expressions with colons (but not already converted)
    content = re.sub(
        r'backend\.at_add\((\w+), ([^,:\)]+):([^,\)]+), ([^)]+)\)',
        convert_expr_slice,
        content
    )

    # Check for module-level backend usage (lines not indented)
    # and add backend = get_backend() after imports if needed
    lines = content.split('\n')
    needs_module_backend = False
    import_end_idx = 0
    in_multiline_import = False

    # First pass: find import section end and check for module-level backend use
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Only consider top-level imports (no leading whitespace)
        is_top_level = not line.startswith(' ') and not line.startswith('\t')

        # Track multi-line imports (parentheses) - only at top level
        if is_top_level and (stripped.startswith('from ') or stripped.startswith('import ')):
            import_end_idx = i
            if '(' in line and ')' not in line:
                in_multiline_import = True

        # Track end of multi-line import
        if in_multiline_import:
            import_end_idx = i
            if ')' in line:
                in_multiline_import = False

        # Check for module-level backend usage (lines not starting with space)
        if 'backend.' in line and is_top_level:
            # Not inside a function/class (no leading whitespace)
            if not stripped.startswith('def ') and not stripped.startswith('class '):
                needs_module_backend = True

    if needs_module_backend:
        # Insert backend = get_backend() after imports
        new_lines = lines[:import_end_idx + 1]
        new_lines.append('')
        new_lines.append('# Get backend at module level')
        new_lines.append('backend = get_backend()')
        new_lines.extend(lines[import_end_idx + 1:])
        content = '\n'.join(new_lines)

    # Ensure 'from typing import Any' is present
    if 'Any' in content:
        if 'from typing import' in content:
            typing_match = re.search(r'from typing import ([^\n]+)', content)
            if typing_match and 'Any' not in typing_match.group(1):
                content = re.sub(
                    r'from typing import ',
                    'from typing import Any, ',
                    content,
                    count=1
                )
        else:
            # Add typing import after the first docstring or at the top
            if '"""' in content:
                # Find end of module docstring
                first_triple = content.find('"""')
                second_triple = content.find('"""', first_triple + 3)
                if second_triple != -1:
                    insert_pos = content.find('\n', second_triple) + 1
                    content = content[:insert_pos] + '\nfrom typing import Any\n' + content[insert_pos:]
            else:
                content = 'from typing import Any\n' + content

    # Add backend = get_backend() at the start of functions/methods that use backend
    lines = content.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Check if this is a function/method definition
        func_match = re.match(r'^(\s*)(def\s+\w+\([^)]*\).*:)\s*$', line)
        if func_match:
            indent = func_match.group(1)
            func_indent = len(indent)

            # Look ahead to see if this function uses 'backend.'
            uses_backend = False
            has_backend_assign = False
            j = i + 1

            # Skip docstring if present
            docstring_depth = 0
            while j < len(lines):
                check_line = lines[j]

                # Check for docstring
                if '"""' in check_line or "'''" in check_line:
                    count = check_line.count('"""') + check_line.count("'''")
                    if docstring_depth == 0 and count >= 2:
                        # Single line docstring
                        j += 1
                        continue
                    elif docstring_depth == 0:
                        docstring_depth = 1
                        j += 1
                        continue
                    elif docstring_depth == 1:
                        docstring_depth = 0
                        j += 1
                        continue

                if docstring_depth > 0:
                    j += 1
                    continue

                # Check if we've left this function
                stripped = check_line.lstrip()
                if stripped and not stripped.startswith('#'):
                    check_indent = len(check_line) - len(check_line.lstrip())
                    if check_indent <= func_indent and stripped:
                        break

                if 'backend.' in check_line:
                    uses_backend = True
                if 'backend = get_backend()' in check_line:
                    has_backend_assign = True
                    break

                j += 1
                if j > i + 100:  # Safety limit
                    break

            # If function uses backend but doesn't have assignment, add it
            if uses_backend and not has_backend_assign:
                # Find where to insert (after docstring)
                j = i + 1
                insert_after = i

                # Skip to after docstring
                if j < len(lines) and ('"""' in lines[j] or "'''" in lines[j]):
                    # Find end of docstring
                    if lines[j].count('"""') >= 2 or lines[j].count("'''") >= 2:
                        # Single line docstring
                        insert_after = j
                    else:
                        # Multi-line docstring
                        j += 1
                        while j < len(lines):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                insert_after = j
                                break
                            j += 1

                # Insert backend = get_backend() after docstring
                if insert_after > i:
                    # Find the right indent (function body indent)
                    body_indent = ' ' * (func_indent + 4)
                    # We need to insert after insert_after, so we track and do it when we get there
                    pass  # This is complex, let's use a simpler approach

        i += 1

    # Simpler approach: just replace inline
    # For each method body that uses backend.*, add backend = get_backend() after docstring

    # Find all def statements and process them
    result_lines = []
    i = 0
    lines = content.split('\n')

    while i < len(lines):
        line = lines[i]
        result_lines.append(line)

        # Match function definition
        func_match = re.match(r'^(\s*)(def\s+\w+\([^)]*\).*:)\s*$', line)
        if func_match:
            func_indent = len(func_match.group(1))
            body_indent = ' ' * (func_indent + 4)

            # Look ahead for docstring and backend usage
            j = i + 1
            docstring_end = None
            uses_backend = False

            # Check for docstring
            if j < len(lines):
                next_line = lines[j].strip()
                if next_line.startswith('"""') or next_line.startswith("'''"):
                    quote = '"""' if next_line.startswith('"""') else "'''"
                    if next_line.count(quote) >= 2:
                        docstring_end = j
                    else:
                        # Multi-line
                        j += 1
                        while j < len(lines):
                            if quote in lines[j]:
                                docstring_end = j
                                break
                            j += 1

            # Check if function body uses backend
            check_start = (docstring_end + 1) if docstring_end else (i + 1)
            for k in range(check_start, min(check_start + 50, len(lines))):
                check_line = lines[k]
                stripped = check_line.lstrip()
                if stripped and not stripped.startswith('#'):
                    check_indent = len(check_line) - len(check_line.lstrip())
                    if check_indent <= func_indent and stripped:
                        break
                if 'backend.' in check_line and 'backend = get_backend()' not in check_line:
                    uses_backend = True
                if 'backend = get_backend()' in check_line:
                    uses_backend = False
                    break

            # Add backend = get_backend() after docstring if needed
            if uses_backend and docstring_end:
                # Copy lines up to and including docstring end
                while i < docstring_end:
                    i += 1
                    result_lines.append(lines[i])
                # Insert backend = get_backend()
                result_lines.append(f'{body_indent}backend = get_backend()')
            elif uses_backend and not docstring_end:
                # No docstring, insert right after def line
                result_lines.append(f'{body_indent}backend = get_backend()')

        i += 1

    return '\n'.join(result_lines)


def migrate_domain(source_dir: Path, dest_dir: Path, domain_name: str):
    """Migrate a complete domain."""
    source = source_dir / domain_name
    dest = dest_dir / domain_name

    if not source.exists():
        print(f"Source not found: {source}")
        return

    dest.mkdir(parents=True, exist_ok=True)

    for py_file in source.glob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        print(f"  Migrating {py_file.name}...")
        content = py_file.read_text()
        migrated = migrate_file(content, depth=3)

        dest_file = dest / py_file.name
        dest_file.write_text(migrated)

    print(f"  Done: {domain_name}")


def migrate_integration(source_dir: Path, dest_dir: Path):
    """Migrate the integration layer (depth 2)."""
    source = source_dir / "integration"
    dest = dest_dir / "integration"

    if not source.exists():
        print(f"Source not found: {source}")
        return

    dest.mkdir(parents=True, exist_ok=True)

    for py_file in source.glob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        print(f"  Migrating {py_file.name}...")
        content = py_file.read_text()
        migrated = migrate_file(content, depth=2)

        dest_file = dest / py_file.name
        dest_file.write_text(migrated)

    print("  Done: integration")


def main():
    source_dir = Path("/Volumes/VIXinSSD/fluxem-domains/fluxem_domains")
    dest_dir = Path("/Volumes/VIXinSSD/FluxEM/fluxem/domains")

    domains = [
        "physics", "chemistry", "biology", "math", "logic",
        "music", "geometry", "graphs", "sets", "number_theory", "data"
    ]

    for domain in domains:
        print(f"Migrating {domain}...")
        migrate_domain(source_dir, dest_dir, domain)

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
