"""
docstring_generator.py — Generate missing docstrings using Claude.

For units that have no docstring (or a stub), this module calls the
Anthropic API to produce a well-formatted NumPy/Google-style docstring
and patches the source file in place.
"""
import re
import ast
from pathlib import Path
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from parser import ParsedUnit


_SYSTEM_PROMPT = """\
You are an expert Python developer. Your job is to write clear, accurate docstrings.
Follow this format exactly (Google style):

\"\"\"
One-line summary of what the function/class does.

Longer description if needed (2-4 sentences max). Explain the algorithm,
design choice, or key behaviour. Omit if the summary is sufficient.

Args:
    param_name: (type) Description.
    param_name: (type) Description.

Returns:
    (type) Description of the return value.

Raises:
    ExceptionType: When this is raised.

Example:
    >>> result = my_function(x=1)
\"\"\"

Return ONLY the docstring content between the triple quotes. No extra text.
"""


def _build_prompt(unit: ParsedUnit) -> str:
    return f"""Write a docstring for this Python {unit.kind}:

File: {unit.file_path}
Module: {unit.module}

```python
{unit.source}
```

Return only the docstring text (without the surrounding triple quotes).
"""


def generate_docstring(unit: ParsedUnit, client: "anthropic.Anthropic") -> Optional[str]:
    """
    Call Claude to generate a docstring for a ParsedUnit that lacks one.

    :param unit: (ParsedUnit) The code unit needing a docstring.
    :param client: (anthropic.Anthropic) Authenticated Anthropic client.
    :return: (str | None) Generated docstring text, or None on failure.
    """
    if not _ANTHROPIC_AVAILABLE:
        raise ImportError("pip install anthropic")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(unit)}],
    )
    text = response.content[0].text.strip()
    # Strip surrounding triple quotes if model included them
    text = re.sub(r'^"""', "", text).strip()
    text = re.sub(r'"""$', "", text).strip()
    return text


def patch_file_with_docstring(unit: ParsedUnit, docstring: str, repo_root: str) -> bool:
    """
    Insert or replace the docstring in the source file for the given unit.

    :param unit: (ParsedUnit) The unit to patch.
    :param docstring: (str) New docstring content (without triple quotes).
    :param repo_root: (str) Repository root to resolve file paths.
    :return: (bool) True if file was modified.
    """
    file_path = Path(repo_root) / unit.file_path
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    # Find the node
    target = None
    for node in ast.walk(tree):
        match unit.kind:
            case "function" | "method":
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = f"{unit.parent}.{node.name}" if unit.parent else node.name
                    if qualified == unit.name and node.lineno == unit.line_start:
                        target = node
            case "class":
                if isinstance(node, ast.ClassDef) and node.name == unit.name:
                    target = node
            case "module":
                target = tree

    if target is None:
        return False

    # Find insertion point: line after def/class line (after colon)
    body = target.body if hasattr(target, "body") else []
    if not body:
        return False

    first_stmt = body[0]
    # If there's already a docstring, replace it
    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
        # Replace existing docstring
        start = first_stmt.lineno - 1
        end = first_stmt.end_lineno
        indent = " " * (first_stmt.col_offset)
        new_doc_lines = f'{indent}"""\n'
        for line in docstring.splitlines():
            new_doc_lines += f"{indent}{line}\n" if line.strip() else "\n"
        new_doc_lines += f'{indent}"""\n'
        lines[start:end] = [new_doc_lines]
    else:
        # Insert before first statement
        insert_at = first_stmt.lineno - 1
        indent = " " * (first_stmt.col_offset)
        new_doc_lines = f'{indent}"""\n'
        for line in docstring.splitlines():
            new_doc_lines += f"{indent}{line}\n" if line.strip() else "\n"
        new_doc_lines += f'{indent}"""\n'
        lines.insert(insert_at, new_doc_lines)

    file_path.write_text("".join(lines), encoding="utf-8")
    return True


def generate_missing_docstrings(
    units: list[ParsedUnit],
    repo_root: str,
    api_key: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """
    Generate docstrings for all units that are missing one, and patch source files.

    :param units: (list[ParsedUnit]) All parsed units.
    :param repo_root: (str) Path to repository root.
    :param api_key: (str | None) Anthropic API key (or set ANTHROPIC_API_KEY env var).
    :param dry_run: (bool) If True, return generated docstrings without writing files.
    :return: (dict) Mapping of unit.id → generated docstring.
    """
    if not _ANTHROPIC_AVAILABLE:
        raise ImportError("pip install anthropic")

    import anthropic as _anthropic
    client = _anthropic.Anthropic(api_key=api_key) if api_key else _anthropic.Anthropic()

    results = {}
    missing = [u for u in units if not u.docstring and u.kind != "module"]
    print(f"Generating docstrings for {len(missing)} units...")

    for unit in missing:
        print(f"  → {unit.module}.{unit.name}")
        try:
            doc = generate_docstring(unit, client)
            results[unit.id] = doc
            if not dry_run:
                patched = patch_file_with_docstring(unit, doc, repo_root)
                status = "patched" if patched else "failed"
                print(f"    {status}")
        except Exception as e:
            print(f"    ERROR: {e}")

    return results
